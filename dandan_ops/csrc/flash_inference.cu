#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <limits>
#include "../../kernels/flash_tensorcore_inference.cuh"

namespace {
// Decode has one query per head. Split the KV sequence into 256-token chunks
// to expose parallelism, then merge the online-softmax states in FP32.
__global__ void decode_d128(
    const half* q, const half* k, const half* v, half* out, float* scratch,
    int Hq, int Hkv, int K, int visible, int splits, float scale) {
    constexpr int D = 128, CHUNK = 256;
    const int row = blockIdx.x, split = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hq = row % Hq, batch = row / Hq, hkv = hq / (Hq / Hkv);
    const size_t kv_base = (size_t(batch) * Hkv + hkv) * K * D;
    const int begin = split * CHUNK, end = min(begin + CHUNK, visible);
    float qf[4], acc[4] = {}, m = -INFINITY, l = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) qf[i] = __half2float(q[size_t(row) * D + lane + 32 * i]);
    for (int token = begin + warp; token < end; token += 4) {
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            score += qf[i] * __half2float(k[kv_base + size_t(token) * D + lane + 32 * i]);
        #pragma unroll
        for (int offset = 16; offset; offset /= 2)
            score += __shfl_xor_sync(0xffffffff, score, offset);
        score *= scale;
        const float next_m = fmaxf(m, score);
        const float correction = expf(m - next_m), p = expf(score - next_m);
        l = l * correction + p;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[i] = acc[i] * correction + p * __half2float(v[kv_base + size_t(token) * D + lane + 32 * i]);
        m = next_m;
    }
    __shared__ float warp_acc[4][D], warp_max[4], warp_sum[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) warp_acc[warp][lane + 32 * i] = acc[i];
    if (lane == 0) { warp_max[warp] = m; warp_sum[warp] = l; }
    __syncthreads();
    float max_all = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 4; ++w) max_all = fmaxf(max_all, warp_max[w]);
    float numerator = 0.0f, denominator = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        const float weight = warp_sum[w] > 0.0f ? expf(warp_max[w] - max_all) : 0.0f;
        numerator += warp_acc[w][threadIdx.x] * weight;
        denominator += warp_sum[w] * weight;
    }
    if (splits == 1) {
        out[size_t(row) * D + threadIdx.x] = __float2half_rn(numerator / denominator);
    } else {
        const size_t off = (size_t(row) * splits + split) * (D + 2);
        scratch[off + threadIdx.x] = numerator;
        if (threadIdx.x == 0) { scratch[off + D] = max_all; scratch[off + D + 1] = denominator; }
    }
}

__global__ void merge_decode(const float* scratch, half* out, int splits) {
    constexpr int D = 128;
    const size_t base = size_t(blockIdx.x) * splits * (D + 2);
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, scratch[base + s * (D + 2) + D]);
    float numerator = 0.0f, denominator = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const size_t off = base + s * (D + 2);
        const float correction = expf(scratch[off + D] - m);
        numerator += scratch[off + threadIdx.x] * correction;
        denominator += scratch[off + D + 1] * correction;
    }
    out[size_t(blockIdx.x) * D + threadIdx.x] = __float2half_rn(numerator / denominator);
}

template<bool CAUSAL>
void launch_prefill(const half* q, const half* k, const half* v, half* out,
    int B, int Hq, int Hkv, int Q, int K, int start, float scale, cudaStream_t stream) {
    constexpr int Br = 64, Bc = 32, D = 128;
    constexpr int smem = (4 * Bc + Br) * (D + 8) * sizeof(half);
    C10_CUDA_CHECK(cudaFuncSetAttribute(dandan_flash::prefill<CAUSAL, Br, Bc, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid((Q + Br - 1) / Br, B * Hq);
    dandan_flash::prefill<CAUSAL, Br, Bc, D><<<grid, 128, smem, stream>>>(
        q, k, v, out, Q, K, Hq, Hkv, start, scale);
}

at::Tensor flash_inference(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, int64_t query_start, bool causal, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "flash_attention expects CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                v.scalar_type() == at::kHalf, "flash_attention expects FP16 tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "flash_attention expects 4D [B,H,N,D] tensors");
    TORCH_CHECK(q.numel() > 0 && k.numel() > 0 && v.numel() > 0,
                "flash_attention dimensions must be non-empty");
    TORCH_CHECK(q.size(0) == k.size(0) && k.sizes() == v.sizes(),
                "K and V must have the same shape and batch as Q");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "query heads must be divisible by KV heads");
    TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128,
                "inference CUDA FlashAttention currently requires head_dim=128");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "flash_attention inputs must be contiguous");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "flash_attention tensors must share a device");
    TORCH_CHECK(q.numel() <= INT_MAX && k.numel() <= INT_MAX && q.size(0) * q.size(1) <= 65535,
                "flash_attention tensor is too large for this kernel");
    TORCH_CHECK(query_start >= 0 && query_start <= k.size(2) - q.size(2),
                "query_start must map Q rows into K rows");
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
                "scale must be finite and representable as FP32");
    TORCH_CHECK(!c10::GradMode::is_enabled() || !(q.requires_grad() || k.requires_grad() || v.requires_grad()),
                "flash_attention CUDA backend is inference-only");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0,
                "flash_attention inputs require 16-byte alignment");
    c10::cuda::CUDAGuard guard(q.device());
    int major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device()));
    TORCH_CHECK(major >= 8, "flash_attention requires SM80+");
    const int B = int(q.size(0)), Hq = int(q.size(1)), Hkv = int(k.size(1));
    const int Q = int(q.size(2)), K = int(k.size(2));
    auto out = at::empty_like(q);
    const auto* qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const auto* kp = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    const auto* vp = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    auto* op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    if (Q == 1) {
        const int visible = causal ? int(query_start) + 1 : K;
        const int splits = (visible + 255) / 256;
        at::Tensor scratch;
        float* sp = nullptr;
        if (splits > 1) {
            scratch = at::empty({B * Hq, splits, 130}, q.options().dtype(at::kFloat));
            sp = scratch.data_ptr<float>();
        }
        decode_d128<<<dim3(B * Hq, splits), 128, 0, stream>>>(
            qp, kp, vp, op, sp, Hq, Hkv, K, visible, splits, float(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (splits > 1) merge_decode<<<B * Hq, 128, 0, stream>>>(sp, op, splits);
    } else if (causal) {
        launch_prefill<true>(qp, kp, vp, op, B, Hq, Hkv, Q, K, int(query_start), float(scale), stream);
    } else {
        launch_prefill<false>(qp, kp, vp, op, B, Hq, Hkv, Q, K, int(query_start), float(scale), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(dandan_ops, m) {
    m.def("flash_inference(Tensor q, Tensor k, Tensor v, int query_start, bool causal, float scale=0.08838834764831845) -> Tensor");
}
TORCH_LIBRARY_IMPL(dandan_ops, CUDA, m) { m.impl("flash_inference", &flash_inference); }
