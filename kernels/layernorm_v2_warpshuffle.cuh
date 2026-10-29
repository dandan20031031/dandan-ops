#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Centered two-pass variance avoids cancellation in E[x*x] - E[x]*E[x].
// Preserve the original one-block-per-row, FP32 inference interface.
__device__ float layernorm_block_sum(float value, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x >> 5) ? scratch[lane] : 0.f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

__global__ void layernorm_v2_warpshuffle(
    const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ beta, float* __restrict__ y, int M, int D, float eps) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const size_t base = static_cast<size_t>(row) * D;
    __shared__ float scratch[32];
    float sum = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) sum += x[base + d];
    const float mean = layernorm_block_sum(sum, scratch) / D;
    // All threads finish reading scratch before the second reduction.
    __syncthreads();
    float squares = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float centered = x[base + d] - mean;
        squares += centered * centered;
    }
    const float inverse = rsqrtf(layernorm_block_sum(squares, scratch) / D + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        y[base + d] = (x[base + d] - mean) * inverse * gamma[d] + beta[d];
}
