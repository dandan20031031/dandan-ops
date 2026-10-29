#pragma once
#include "flash_tensorcore_inference.cuh"

namespace dandan_flash {
// Split KV into 256-token chunks. Four warps scan disjoint tokens and
// combine FP32 online-softmax states without materializing scores.
template<typename T, int D, bool PAGED>
__global__ void decode(const T* q, const T* k, const T* v, T* out, float* scratch,
    int Hq, int Hkv, int K, int visible, int splits, float scale, int window,
    float softcap, const int* tables, const int* lengths, int width, int block_size,
    int num_blocks) {
    constexpr int CHUNK = 256, R = D / 32;
    const int row = blockIdx.x, split = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hq = row % Hq, batch = row / Hq, hkv = hq / (Hq / Hkv);
    if constexpr (PAGED) {
        visible = lengths[batch];
        if (visible < 0 || visible > K || visible > int64_t(width) * block_size) visible = 0;
    }
    const int lower = window > 0 ? max(0, visible - window) : 0;
    const int begin = max(split * CHUNK, lower), end = min((split + 1) * CHUNK, visible);
    float qf[R], acc[R] = {}, m = -INFINITY, l = 0.0f;
    #pragma unroll
    for (int i = 0; i < R; ++i) qf[i] = as_float(q[size_t(row) * D + lane + 32 * i]);
    for (int token = begin + warp; token < end; token += 4) {
        size_t base;
        if constexpr (PAGED) {
            const int page = tables[size_t(batch) * width + token / block_size];
            if (page < 0 || page >= num_blocks) continue;
            base = ((size_t(page) * block_size + token % block_size) * Hkv + hkv) * D;
        } else base = ((size_t(batch) * Hkv + hkv) * K + token) * D;
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < R; ++i) score += qf[i] * as_float(k[base + lane + 32 * i]);
        #pragma unroll
        for (int offset = 16; offset; offset /= 2) score += __shfl_xor_sync(0xffffffff, score, offset);
        score *= scale;
        if (softcap > 0.0f) score = softcap * tanhf(score / softcap);
        const float next_m = fmaxf(m, score);
        const float correction = expf(m - next_m), p = expf(score - next_m);
        l = l * correction + p;
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = acc[i] * correction + p * as_float(v[base + lane + 32 * i]);
        m = next_m;
    }
    __shared__ float warp_acc[4][D], warp_max[4], warp_sum[4];
    #pragma unroll
    for (int i = 0; i < R; ++i) warp_acc[warp][lane + 32 * i] = acc[i];
    if (lane == 0) { warp_max[warp] = m; warp_sum[warp] = l; }
    __syncthreads();
    float max_all = -INFINITY, denominator = 0.0f;
    float weights[4];
    #pragma unroll
    for (int w = 0; w < 4; ++w) max_all = fmaxf(max_all, warp_max[w]);
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        weights[w] = warp_sum[w] > 0.0f ? expf(warp_max[w] - max_all) : 0.0f;
        denominator += warp_sum[w] * weights[w];
    }
    const size_t off = (size_t(row) * splits + split) * (D + 2);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float numerator = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) numerator += warp_acc[w][d] * weights[w];
        if (splits == 1) out[size_t(row) * D + d] = from_float<T>(denominator > 0 ? numerator / denominator : 0.0f);
        else scratch[off + d] = numerator;
    }
    if (splits > 1 && threadIdx.x == 0) {
        scratch[off + D] = max_all; scratch[off + D + 1] = denominator;
    }
}

template<typename T, int D>
__global__ void merge_decode(const float* scratch, T* out, int splits) {
    const size_t base = size_t(blockIdx.x) * splits * (D + 2);
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, scratch[base + s * (D + 2) + D]);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float numerator = 0.0f, denominator = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const size_t off = base + s * (D + 2);
            const float correction = scratch[off + D + 1] > 0 ? expf(scratch[off + D] - m) : 0.0f;
            numerator += scratch[off + d] * correction;
            denominator += scratch[off + D + 1] * correction;
        }
        out[size_t(blockIdx.x) * D + d] = from_float<T>(denominator > 0 ? numerator / denominator : 0.0f);
    }
}
} // namespace dandan_flash
