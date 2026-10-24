#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace dandan_w4 {
// A warp owns an output column; lanes load consecutive packed words along K.
// Each word and scale/zero is reused across up to eight activation rows.
// Eight lanes share one 64-element quantization group. No CTA-wide reduction
// or complete dequantized weight tensor is needed. FP16 weight rounding is
// identical to the Tensor Core path, followed by FP32 products/sums.
template<int ROWS>
__global__ void __launch_bounds__(128) small_batch(
    const half* x, const int32_t* packed, const float* scales,
    const int32_t* zeros, half* out, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (col >= N) return;
    const int words = (K + 7) / 8, groups = (K + 63) / 64;
    // For M=1 preserve the original 256-thread GEMV addition tree exactly.
    // Each lane represents eight adjacent old threads. This keeps cached
    // generation numerically stable while removing CTA-wide barriers.
    constexpr int PARTS = ROWS == 1 ? 8 : 1;
    float sums[ROWS][PARTS] = {};
    for (int word = lane; word < ((words + 31) / 32) * 32; word += 32) {
        const int k0 = word * 8;
        const uint32_t bits = word < words ? uint32_t(packed[int64_t(col) * words + word]) : 0;
        float scale = 0.0f;
        int zero = 0;
        if ((lane & 7) == 0 && k0 < K) {
            scale = scales[int64_t(col) * groups + k0 / 64];
            zero = zeros[int64_t(col) * groups + k0 / 64];
        }
        scale = __shfl_sync(0xffffffff, scale, lane & ~7);
        zero = __shfl_sync(0xffffffff, zero, lane & ~7);
        float weights[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            weights[j] = __half2float(__float2half_rn((int((bits >> (4*j)) & 15)-zero) * scale));
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < M) {
                if (K % 8 == 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0) {
                    if (k0 < K) {
                        const uint4 vec = *reinterpret_cast<const uint4*>(x + int64_t(r)*K+k0);
                        const uint32_t pairs[4] = {vec.x, vec.y, vec.z, vec.w};
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            sums[r][ROWS == 1 ? 2*j : 0] += __half2float(__ushort_as_half(pairs[j] & 0xffff)) * weights[2*j];
                            sums[r][ROWS == 1 ? 2*j+1 : 0] += __half2float(__ushort_as_half(pairs[j] >> 16)) * weights[2*j+1];
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        if (k0+j < K) sums[r][ROWS == 1 ? j : 0] += __half2float(x[int64_t(r)*K+k0+j]) * weights[j];
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int offset = 16; offset; offset /= 2)
            #pragma unroll
            for (int part = 0; part < PARTS; ++part)
                sums[r][part] += __shfl_down_sync(0xffffffff, sums[r][part], offset);
        if constexpr (ROWS == 1) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) sums[r][j] += sums[r][j+4];
            sums[r][0] += sums[r][2]; sums[r][1] += sums[r][3];
            sums[r][0] += sums[r][1];
        }
        if (lane == 0 && r < M) out[int64_t(r) * N + col] = __float2half_rn(sums[r][0]);
    }
}
} // namespace dandan_w4
