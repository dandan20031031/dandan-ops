// batchnorm_v2_rowblock.cuh    BatchNorm    V2 row-block     
//
//    v1       
//   1)    i % D   64                
//   2)      block col   threadIdx.x        channel =    
//         coalesced a/b            
//
// BN           kernel       block    /sync 
//               LN/RMSNorm  "    block"     
//            

#include <cuda_runtime.h>

__global__ void batchnorm_v2_rowblock(
    const float* __restrict__ x,   // [M, D] row-major
    const float* __restrict__ a,   // [D]     
    const float* __restrict__ b,   // [D]     
    float* __restrict__ y,         // [M, D] row-major
    int M, int D)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const float* x_row = x + (size_t)row * D;
    float* y_row = y + (size_t)row * D;

    //        x   col    a/b   col   y   
    for (int channel_index = threadIdx.x; channel_index < D; channel_index += blockDim.x) {
        y_row[channel_index] = x_row[channel_index] * a[channel_index] + b[channel_index];
    }
}
