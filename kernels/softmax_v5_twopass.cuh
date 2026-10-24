#pragma once
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_v5_twopass(float const* input, float* output, int M, int N) {
    int row = (int)blockIdx.x;
    if (row >= M) return;
    float max_value = -INFINITY;
    for (int i = (int)threadIdx.x; i < N; i += (int)blockDim.x)
        max_value = fmaxf(max_value, input[row * N + i]);
    __shared__ float max_values[256];
    max_values[threadIdx.x] = max_value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            max_values[threadIdx.x] = fmaxf(max_values[threadIdx.x], max_values[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = max_values[0];
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < N; i += (int)blockDim.x)
        sum += expf(input[row * N + i] - row_max);
    __shared__ float sums[256];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = sums[0];
    for (int i = (int)threadIdx.x; i < N; i += (int)blockDim.x)
        output[row * N + i] = expf(input[row * N + i] - row_max) / row_sum;
}
