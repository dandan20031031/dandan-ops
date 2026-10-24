// rmsnorm_v3_float4.cuh    RMSNorm V3 float4 128-bit, LDG.128      
//
//   v2       pass1/pass2     strided     float4     
//      v2   warp shuffle            A/B            
//
//     float4           LDG.128   16    4   float  
//      LDG.32 4           1/4              4 
//     memory-level parallelism    memory-bound            
//
//    D % 4 == 0     16B    float4       LLM hidden 4096/5120/8192     
//   4      D        /     launch    assert    

#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_v3_float4(
    const float* __restrict__ x,      // [M, D] row-major D % 4 == 0
    const float* __restrict__ gamma,  // [D] D % 4 == 0
    float* __restrict__ y,            // [M, D] row-major
    int M, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    //      float4     x    16B    cudaMalloc  D%4==0        16B   
    const float4* __restrict__ x_vec = (const float4*)(x + (size_t)row * D);
    float4* __restrict__ y_vec = (float4*)(y + (size_t)row * D);
    const float4* __restrict__ g_vec = (const float4*)gamma;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int num_vecs = D >> 2;          //        float4

    // ---- pass 1: float4   x    4         ----
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < num_vecs; v += blockDim.x) {
        const float4 xv = x_vec[v];
        local_sum += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }

    // ---- stage 1: warp         v2    ----
    float warp_sum = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
    }

    // ---- stage 2: warp       v2    ----
    __shared__ float s_warp_sum[32];
    if (lane == 0) s_warp_sum[warp] = warp_sum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? s_warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) s_warp_sum[0] = v;
    }
    __syncthreads();

    const float scale = rsqrtf(s_warp_sum[0] / (float)D + eps);

    // ---- pass 2: float4   x   gamma    float4   y ----
    for (int v = threadIdx.x; v < num_vecs; v += blockDim.x) {
        const float4 xv = x_vec[v];
        const float4 gv = g_vec[v];
        float4 yv;
        yv.x = xv.x * scale * gv.x;
        yv.y = xv.y * scale * gv.y;
        yv.z = xv.z * scale * gv.z;
        yv.w = xv.w * scale * gv.w;
        y_vec[v] = yv;
    }
}
