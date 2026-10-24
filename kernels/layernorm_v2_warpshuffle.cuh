// layernorm_v2_warpshuffle.cuh    LayerNorm V2 warp shuffle      
//
//   v1            sum   sumsq             
//      __shfl_xor_sync    2   shuffle         rmsnorm_v2 
//        strided    A/B             

#include <cuda_runtime.h>
#include <math.h>

__global__ void layernorm_v2_warpshuffle(
    const float* __restrict__ x,      // [M, D] row-major
    const float* __restrict__ gamma,  // [D]
    const float* __restrict__ beta,   // [D]
    float* __restrict__ y,            // [M, D] row-major
    int M, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const float* x_row = x + (size_t)row * D;
    float* y_row = y + (size_t)row * D;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // ---- pass 1:       x    x  ----
    float sum = 0.0f, sumsq = 0.0f;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        const float v = x_row[col];
        sum += v;
        sumsq += v * v;
    }

    // ---- stage 1: warp       sum/sumsq          ----
    float wsum = sum, wsumsq = sumsq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        wsum   += __shfl_xor_sync(0xffffffff, wsum, offset);
        wsumsq += __shfl_xor_sync(0xffffffff, wsumsq, offset);
    }

    // ---- stage 2: warp     ----
    __shared__ float s_wsum[32];
    __shared__ float s_wsumsq[32];
    if (lane == 0) {
        s_wsum[warp] = wsum;
        s_wsumsq[warp] = wsumsq;
    }
    __syncthreads();

    if (warp == 0) {
        const int nwarps = blockDim.x >> 5;
        float v1 = (lane < nwarps) ? s_wsum[lane] : 0.0f;
        float v2 = (lane < nwarps) ? s_wsumsq[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v1 += __shfl_xor_sync(0xffffffff, v1, offset);
            v2 += __shfl_xor_sync(0xffffffff, v2, offset);
        }
        if (lane == 0) {
            s_wsum[0] = v1;
            s_wsumsq[0] = v2;
        }
    }
    __syncthreads();

    const float mean = s_wsum[0] / (float)D;
    const float var  = s_wsumsq[0] / (float)D - mean * mean;
    const float rstd = rsqrtf(var + eps);

    // ---- pass 2: (x -  ) * rstd *   +   ----
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        const float v = x_row[col];
        y_row[col] = (v - mean) * rstd * gamma[col] + beta[col];
    }
}
