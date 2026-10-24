#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Derived from cuda_code/hgemm/hgemm_v8_sm120_mmasync.cu.
// Retains its 128-byte swizzle, ldmatrix fragment mapping, m16n8k16 MMA,
// FP32 accumulators and two-stage cp.async pipeline. Additions: masked tails,
// small-M tile specializations, FP16 stores and packed INT4 tile loading.
namespace dandan_hgemm {

__device__ __forceinline__ int swizzle(int row, int vector) {
    return row * 128 + (vector ^ (row & 7)) * 16;
}

__device__ __forceinline__ void ldmatrix(int (&reg)[4], int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
}

__device__ __forceinline__ void mma(const int* a, const int* b, float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template<int HEIGHT, int THREADS>
__device__ __forceinline__ void load_half(
    int dst, const half* src, int rows, int K, int row0, int k0, int tid) {
    for (int index = tid * 8; index < HEIGHT * 64; index += THREADS * 8) {
        const int row = index / 64, col = index % 64;
        const int gr = row0 + row, gk = k0 + col;
        const int addr = dst + swizzle(row, col / 8);
        if (K % 8 == 0 && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            // cp.async zfill for complete invalid vectors. Do not form an OOB
            // source pointer: the source is unused when source_bytes is zero.
            const bool valid = gr < rows && gk < K;
            const half* ptr = valid ? src + int64_t(gr) * K + gk : src;
            const int source_bytes = valid ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                :: "r"(addr), "l"(ptr), "r"(source_bytes) : "memory");
        } else {
            half* ptr = reinterpret_cast<half*>(__cvta_shared_to_generic(addr));
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                ptr[j] = (gr < rows && gk + j < K)
                    ? src[int64_t(gr) * K + gk + j] : __float2half_rn(0.0f);
        }
    }
}

template<int THREADS>
__device__ __forceinline__ void load_int4(
    int dst, const int32_t* src, const float* scales, const int32_t* zeros,
    int N, int K, int row0, int k0, int tid) {
    const int words = (K + 7) / 8, groups = (K + 63) / 64;
    for (int index = tid; index < 64 * 8; index += THREADS) {
        const int row = index / 8, vector = index % 8;
        const int gn = row0 + row, gk = k0 + vector * 8;
        uint32_t word = 0;
        float scale = 0;
        int zero = 0;
        if (gn < N && gk < K) {
            word = static_cast<uint32_t>(src[int64_t(gn) * words + gk / 8]);
            scale = scales[int64_t(gn) * groups + gk / 64];
            zero = zeros[int64_t(gn) * groups + gk / 64];
        }
        uint32_t values[4];
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int j = pair * 2;
            const float lo = (gn < N && gk + j < K)
                ? (int((word >> (4 * j)) & 15) - zero) * scale : 0.0f;
            const float hi = (gn < N && gk + j + 1 < K)
                ? (int((word >> (4 * (j + 1))) & 15) - zero) * scale : 0.0f;
            values[pair] = uint32_t(__half_as_ushort(__float2half_rn(lo)))
                | (uint32_t(__half_as_ushort(__float2half_rn(hi))) << 16);
        }
        auto* ptr = reinterpret_cast<uint4*>(__cvta_shared_to_generic(dst + swizzle(row, vector)));
        *ptr = make_uint4(values[0], values[1], values[2], values[3]);
    }
}

template<bool QUANTIZED, int BM>
__global__ __launch_bounds__((BM / 16) * (BM <= 32 ? 4 : 1) * 32)
void pipeline_kernel(const half* A, const void* B, const float* scales,
                     const int32_t* zeros, half* C, int M, int N, int K) {
    constexpr int BN = 64, BK = 64, WM = 16;
    constexpr int WARPS_N = BM <= 32 ? 4 : 1;
    constexpr int WN = BN / WARPS_N;
    constexpr int THREADS = (BM / 16) * WARPS_N * 32;
    constexpr int STAGE_BYTES = (BM + BN) * BK * sizeof(half);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
    const int grid_n = (N + BN - 1) / BN;
    const int off_m = (blockIdx.x / grid_n) * BM;
    const int off_n = (blockIdx.x % grid_n) * BN;
    extern __shared__ __align__(16) half dandan_pipeline_smem[];
    const int base = int(__cvta_generic_to_shared(dandan_pipeline_smem));
    const int a_shared = base, b_shared = base + BM * BK * sizeof(half);
    const int a_thread = a_shared + swizzle(warp_m * WM + lane % 16, lane / 16);
    const int b_thread = b_shared + swizzle(warp_n * WN + lane % 8 + (lane / 16) * 8, (lane % 16) / 8);
    float acc[WN / 8][4] = {};
    int af[BK / 16][4], bf[BK / 16][WN / 8][2];
    const int stages = (K + BK - 1) / BK;

    auto load = [&](int stage) {
        int offset = (stage & 1) * STAGE_BYTES;
        load_half<BM, THREADS>(a_shared + offset, A, M, K, off_m, stage * BK, tid);
        if constexpr (QUANTIZED)
            load_int4<THREADS>(b_shared + offset, static_cast<const int32_t*>(B),
                              scales, zeros, N, K, off_n, stage * BK, tid);
        else
            load_half<BN, THREADS>(b_shared + offset, static_cast<const half*>(B),
                                  N, K, off_n, stage * BK, tid);
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    load(0);
    for (int stage = 0; stage < stages; ++stage) {
        if (stage + 1 < stages) load(stage + 1);
        else asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();
        const int offset = (stage & 1) * STAGE_BYTES;
        #pragma unroll
        for (int k = 0; k < BK / 16; ++k) {
            ldmatrix(af[k], (a_thread + offset) ^ (k * 32));
            #pragma unroll
            for (int n = 0; n < WN / 8; n += 2) {
                // The original writes four registers through a two-register
                // subarray. Keep its mapping, with explicit bounds here.
                int regs[4];
                ldmatrix(regs, (b_thread + offset + n * 8 * BK * sizeof(half)) ^ (k * 32));
                bf[k][n][0] = regs[0]; bf[k][n][1] = regs[1];
                bf[k][n + 1][0] = regs[2]; bf[k][n + 1][1] = regs[3];
            }
        }
        #pragma unroll
        for (int k = 0; k < BK / 16; ++k)
            #pragma unroll
            for (int n = 0; n < WN / 8; ++n) mma(af[k], bf[k][n], acc[n]);
        __syncthreads(); // all readers finished before this stage can be reused
    }
    #pragma unroll
    for (int n = 0; n < WN / 8; ++n) {
        const int row = off_m + warp_m * WM + lane / 4;
        const int col = off_n + warp_n * WN + n * 8 + (lane % 4) * 2;
        if (row < M && col < N) C[int64_t(row) * N + col] = __float2half_rn(acc[n][0]);
        if (row < M && col + 1 < N) C[int64_t(row) * N + col + 1] = __float2half_rn(acc[n][1]);
        if (row + 8 < M && col < N) C[int64_t(row + 8) * N + col] = __float2half_rn(acc[n][2]);
        if (row + 8 < M && col + 1 < N) C[int64_t(row + 8) * N + col + 1] = __float2half_rn(acc[n][3]);
    }
}
} // namespace dandan_hgemm
