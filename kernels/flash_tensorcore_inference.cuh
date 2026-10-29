#pragma once
// Inference adaptation of cuda_code/flashAttention/flashattention_v2_1.cuh.
// Original SHA256: 49f11c12eb636ea2d23076b79a14b481814c6ca87a5aff760401d357330f6ae4
// Keeps the original ldmatrix/MMA fragment mapping and online softmax.
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <cstdint>
#include <math.h>

namespace dandan_flash {

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(address));
}

template<typename T>
__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    if constexpr (std::is_same<T, half>::value) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

__device__ __forceinline__ uint32_t shared_address(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template<typename T> __device__ __forceinline__ T from_float(float value);
template<> __device__ __forceinline__ half from_float<half>(float value) { return __float2half_rn(value); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) { return __float2bfloat16_rn(value); }
template<typename T> __device__ __forceinline__ float as_float(T value);
template<> __device__ __forceinline__ float as_float<half>(half value) { return __half2float(value); }
template<> __device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template<typename T>
__device__ __forceinline__ uint32_t pack_pair(float lo, float hi) {
    if constexpr (std::is_same<T, half>::value)
        return uint32_t(__half_as_ushort(__float2half_rn(lo))) |
               (uint32_t(__half_as_ushort(__float2half_rn(hi))) << 16);
    else
        return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
               (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

template<int Bc, int D, typename T>
__device__ __forceinline__ void copy_kv(
    const T* k, const T* v, T* ks, T* vs,
    size_t base, int start, int length, int buffer, size_t stride) {
    constexpr int DP = D + 8;
    constexpr int PACKS = D / 8;
    for (int i = threadIdx.x; i < Bc * PACKS; i += blockDim.x) {
        const int row = i / PACKS, col = (i % PACKS) * 8;
        const bool valid = start + row < length;
        // Use an in-bounds address even for zero-fill cp.async.
        const T* kg = valid ? k + base + size_t(start + row) * stride + col : k;
        const T* vg = valid ? v + base + size_t(start + row) * stride + col : v;
        T* kd = ks + size_t(buffer) * Bc * DP + row * DP + col;
        T* vd = vs + size_t(buffer) * Bc * DP + row * DP + col;
        __pipeline_memcpy_async(kd, kg, 16, valid ? 0 : 16);
        __pipeline_memcpy_async(vd, vg, 16, valid ? 0 : 16);
    }
}

// Four warps, 16 query rows per warp. D=128 uses fewer accumulator registers
// than the old 32-rows-per-warp D=64 experiment. Q stays in registers; K/V
// uses two shared-memory buffers. No [Q,K] score tensor is materialized.
template<bool CAUSAL, int Br = 64, int Bc = 32, int D = 128, typename T = half, bool VARLEN = false>
__global__ void __launch_bounds__(128) prefill(
    const T* q, const T* k, const T* v, T* out,
    int Q, int K, int Hq, int Hkv, int query_start, float scale,
    int window_size, float softcap, const int* cu_q = nullptr,
    const int* cu_k = nullptr, const int* starts = nullptr, int total_q = 0) {
    static_assert(Br == 64 && Bc % 16 == 0 && D % 16 == 0, "invalid flash tile");
    constexpr int DP = D + 8, KM = D / 16, NM = Bc / 8;
    constexpr int KP = Bc / 16, ND = D / 8;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane / 4, t = lane % 4, tl = lane / 8, tr = lane % 8;
    const int q_start = blockIdx.x * Br;
    const int bh = blockIdx.y;
    const int batch = bh / Hq, hq = bh % Hq, hkv = hq / (Hq / Hkv);
    size_t q_base, kv_base;
    size_t q_stride = D, kv_stride = D;
    if constexpr (VARLEN) {
        // High-level plans validate values on CPU. Bound raw device metadata
        // too, so malformed offsets cannot issue out-of-range memory reads.
        if (cu_q[batch] < 0 || cu_q[batch+1] < cu_q[batch] || cu_q[batch+1] > total_q ||
            cu_k[batch] < 0 || cu_k[batch+1] < cu_k[batch] || cu_k[batch+1] > K) return;
        Q = cu_q[batch + 1] - cu_q[batch];
        K = cu_k[batch + 1] - cu_k[batch];
        query_start = starts[batch];
        if (query_start < 0 || query_start > K - Q) return;
        q_base = size_t(cu_q[batch]) * Hq * D + hq * D;
        kv_base = size_t(cu_k[batch]) * Hkv * D + hkv * D;
        q_stride = size_t(Hq) * D; kv_stride = size_t(Hkv) * D;
    } else {
        q_base = size_t(bh) * Q * D;
        kv_base = (size_t(batch) * Hkv + hkv) * K * D;
    }
    if (q_start >= Q) return;
    const int limit = CAUSAL ? min(K, query_start + min(q_start + Br, Q)) : K;
    const int first_tile = CAUSAL && window_size > 0
        ? max(0, query_start + q_start - window_size + 1) / Bc : 0;
    const int tiles = (limit + Bc - 1) / Bc - first_tile;
    const float sl2e = scale * 1.4426950408889634f;

    extern __shared__ __align__(16) unsigned char flash_smem_raw[];
    T* ks = reinterpret_cast<T*>(flash_smem_raw);
    T* vs = ks + 2 * Bc * DP;
    T* qs = vs + 2 * Bc * DP;
    for (int i = threadIdx.x; i < Br * (D / 8); i += blockDim.x) {
        const int row = i / (D / 8), col = (i % (D / 8)) * 8;
        uint4* dst = reinterpret_cast<uint4*>(qs + row * DP + col);
        *dst = q_start + row < Q
            ? *reinterpret_cast<const uint4*>(q + q_base + size_t(q_start + row) * q_stride + col)
            : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    uint32_t qf[KM][4];
    #pragma unroll
    for (int kk = 0; kk < KM; ++kk) {
        const int row = warp * 16 + tr + ((tl & 1) ? 8 : 0);
        const int col = kk * 16 + ((tl & 2) ? 8 : 0);
        ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                    shared_address(qs + row * DP + col));
    }
    float running_max[2] = {-INFINITY, -INFINITY};
    float partial_sum[2] = {0.0f, 0.0f};
    float acc[ND][4] = {};

    copy_kv<Bc, D, T>(k, v, ks, vs, kv_base, first_tile * Bc, K, 0, kv_stride);
    __pipeline_commit();
    for (int j = 0; j < tiles; ++j) {
        const int start = (j + first_tile) * Bc, buffer = j & 1;
        if (j + 1 < tiles) {
            copy_kv<Bc, D, T>(k, v, ks, vs, kv_base, start + Bc, K, buffer ^ 1, kv_stride);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        T* kj = ks + size_t(buffer) * Bc * DP;
        T* vj = vs + size_t(buffer) * Bc * DP;
        uint32_t kf[KM][NM][2];
        #pragma unroll
        for (int kk = 0; kk < KM; ++kk)
            #pragma unroll
            for (int p = 0; p < NM / 2; ++p) {
                const int row = p * 16 + tr + ((tl >> 1) ? 8 : 0);
                const int col = kk * 16 + ((tl & 1) ? 8 : 0);
                ldmatrix_x4(kf[kk][2*p][0], kf[kk][2*p][1],
                    kf[kk][2*p+1][0], kf[kk][2*p+1][1], shared_address(kj + row * DP + col));
            }
        float scores[NM][4] = {};
        #pragma unroll
        for (int ni = 0; ni < NM; ++ni)
            #pragma unroll
            for (int kk = 0; kk < KM; ++kk)
                mma_m16n8k16<T>(scores[ni][0], scores[ni][1], scores[ni][2], scores[ni][3],
                    qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], kf[kk][ni][0], kf[kk][ni][1]);

        // Full blocks strictly before every query row need only scale.
        const bool diagonal = CAUSAL && start + Bc - 1 > query_start + q_start;
        #pragma unroll
        for (int ni = 0; ni < NM; ++ni)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = q_start + warp * 16 + g + 8 * h;
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int col = start + ni * 8 + 2 * t + c;
                    const bool masked = col >= K || (diagonal && col > query_start + row)
                        || (window_size > 0 && col < query_start + row - window_size + 1);
                    const float raw = scores[ni][2*h+c];
                    const float logit = softcap > 0.0f
                        ? softcap * tanhf(raw * scale / softcap) * 1.4426950408889634f
                        : raw * sl2e;
                    scores[ni][2*h+c] = masked ? -INFINITY : logit;
                }
            }
        float next_max[2], correction[2], max_for_exp[2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = -INFINITY;
            #pragma unroll
            for (int ni = 0; ni < NM; ++ni)
                m = fmaxf(m, fmaxf(scores[ni][2*h], scores[ni][2*h+1]));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
            next_max[h] = fmaxf(running_max[h], m);
            max_for_exp[h] = next_max[h] == -INFINITY ? 0.0f : next_max[h];
            correction[h] = exp2f(running_max[h] - max_for_exp[h]);
            partial_sum[h] *= correction[h];
            #pragma unroll
            for (int ni = 0; ni < NM; ++ni) {
                scores[ni][2*h] = exp2f(scores[ni][2*h] - max_for_exp[h]);
                scores[ni][2*h+1] = exp2f(scores[ni][2*h+1] - max_for_exp[h]);
                partial_sum[h] += scores[ni][2*h] + scores[ni][2*h+1];
            }
        }
        uint32_t pf[KP][4];
        #pragma unroll
        for (int kk = 0; kk < KP; ++kk) {
            pf[kk][0] = pack_pair<T>(scores[2*kk][0], scores[2*kk][1]);
            pf[kk][1] = pack_pair<T>(scores[2*kk][2], scores[2*kk][3]);
            pf[kk][2] = pack_pair<T>(scores[2*kk+1][0], scores[2*kk+1][1]);
            pf[kk][3] = pack_pair<T>(scores[2*kk+1][2], scores[2*kk+1][3]);
        }
        #pragma unroll
        for (int nn = 0; nn < ND; ++nn) {
            acc[nn][0] *= correction[0]; acc[nn][1] *= correction[0];
            acc[nn][2] *= correction[1]; acc[nn][3] *= correction[1];
        }
        uint32_t vf[KP][ND][2];
        #pragma unroll
        for (int kk = 0; kk < KP; ++kk)
            #pragma unroll
            for (int p = 0; p < ND / 2; ++p) {
                const int row = kk * 16 + tr + ((tl & 1) ? 8 : 0);
                const int col = p * 16 + ((tl >> 1) ? 8 : 0);
                ldmatrix_x4_trans(vf[kk][2*p][0], vf[kk][2*p][1],
                    vf[kk][2*p+1][0], vf[kk][2*p+1][1], shared_address(vj + row * DP + col));
            }
        #pragma unroll
        for (int nn = 0; nn < ND; ++nn)
            #pragma unroll
            for (int kk = 0; kk < KP; ++kk)
                mma_m16n8k16<T>(acc[nn][0], acc[nn][1], acc[nn][2], acc[nn][3],
                    pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vf[kk][nn][0], vf[kk][nn][1]);
        running_max[0] = next_max[0]; running_max[1] = next_max[1];
        __syncthreads();
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        float sum = partial_sum[h];
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const int row = q_start + warp * 16 + g + 8 * h;
        if (row < Q) {
            #pragma unroll
            for (int nn = 0; nn < ND; ++nn) {
                const size_t off = q_base + size_t(row) * q_stride + nn * 8 + 2 * t;
                out[off] = from_float<T>(acc[nn][2*h] * inv);
                out[off+1] = from_float<T>(acc[nn][2*h+1] * inv);
            }
        }
    }
}
} // namespace dandan_flash
