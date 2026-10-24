#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "gemm_checks.h"

// The following are the final kernels selected from the learning tree.
#include "../../kernels/softmax_v5_twopass.cuh"
#include "../../kernels/rmsnorm_v3_float4.cuh"
#include "../../kernels/layernorm_v2_warpshuffle.cuh"
#include "../../kernels/batchnorm_v2_rowblock.cuh"
#if defined(DANDAN_ENABLE_FLASHATTENTION)
#include "../../kernels/flashattention_v2_1.cuh"
#endif

namespace {

at::Tensor gemm(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "gemm expects CUDA tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
                "gemm expects FP16 tensors");
    TORCH_CHECK(a.size(1) == b.size(0), "incompatible GEMM dimensions");
    TORCH_CHECK(a.device() == b.device(), "gemm tensors must be on the same device");
    c10::cuda::CUDAGuard guard(a.device());
    // This is the stable public backend. The architecture-specific HGEMM files
    // remain available for profiling and can replace this body without changing
    // the Python API.
    return at::matmul(a.contiguous(), b.contiguous());
}

__global__ void w4a16_gemm_kernel(
    const half* x, const int32_t* packed_w, const float* scales,
    const int32_t* zeros, half* y, int M, int N, int K, int group_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    int words = (K + 7) / 8;
    for (int word = 0; word < words; ++word) {
        uint32_t bits = static_cast<uint32_t>(packed_w[col * words + word]);
        int k0 = word * 8;
        for (int j = 0; j < 8; ++j) {
            int k = k0 + j;
            if (k >= K) break;
            int group = k / group_size;
            int q = static_cast<int>((bits >> (4 * j)) & 0xF);
            float w = (q - zeros[col * ((K + group_size - 1) / group_size) + group])
                    * scales[col * ((K + group_size - 1) / group_size) + group];
            acc += __half2float(x[row * K + k]) * __half2float(__float2half_rn(w));
        }
    }
    y[row * N + col] = __float2half(acc);
}

__global__ void w4a16_gemv_kernel(
    const half* x, const int32_t* packed_w, const float* scales,
    const int32_t* zeros, half* y, int N, int K, int group_size) {
    int out = (int)blockIdx.x;
    if (out >= N) return;
    int groups = (K + group_size - 1) / group_size;
    int words = (K + 7) / 8;
    float value = 0.0f;
    for (int k = (int)threadIdx.x; k < K; k += (int)blockDim.x) {
        int word = k / 8;
        int nibble = k & 7;
        uint32_t bits = static_cast<uint32_t>(packed_w[out * words + word]);
        int q = static_cast<int>((bits >> (4 * nibble)) & 0xF);
        int group = k / group_size;
        float w = (q - zeros[out * groups + group]) * scales[out * groups + group];
        value += __half2float(x[k]) * __half2float(__float2half_rn(w));
    }
    __shared__ float partials[256];
    partials[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[out] = __float2half(partials[0]);
}

at::Tensor w4a16_gemm(const at::Tensor& x, const at::Tensor& packed_w,
                      const at::Tensor& scales, const at::Tensor& zeros,
                      int64_t group_size) {
    check_w4a16(x, packed_w, scales, zeros, group_size);
    TORCH_CHECK(x.is_cuda() && packed_w.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
                "w4a16_gemm expects CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kHalf && packed_w.scalar_type() == at::kInt &&
                    scales.scalar_type() == at::kFloat && zeros.scalar_type() == at::kInt,
                "invalid W4A16 dtypes");
    TORCH_CHECK(x.dim() == 2 && packed_w.dim() == 2 && scales.dim() == 2 && zeros.dim() == 2,
                "W4A16 tensors must be 2D");
    int64_t M = x.size(0), K = x.size(1), N = packed_w.size(0);
    TORCH_CHECK(group_size > 0 && packed_w.size(1) == (K + 7) / 8,
                "invalid W4A16 shape");
    int64_t groups = (K + group_size - 1) / group_size;
    TORCH_CHECK(scales.sizes() == at::IntArrayRef({N, groups}) && zeros.sizes() == scales.sizes(),
                "invalid W4A16 metadata shape");
    TORCH_CHECK(x.device() == packed_w.device() && x.device() == scales.device() &&
                    x.device() == zeros.device(), "W4A16 tensors must share a device");
    c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({M, N}, x.options());
    if (M == 0) return y;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    w4a16_gemm_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), packed_w.data_ptr<int32_t>(),
        scales.data_ptr<float>(), zeros.data_ptr<int32_t>(), reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), static_cast<int>(group_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor w4a16_gemv(const at::Tensor& x, const at::Tensor& packed_w,
                      const at::Tensor& scales, const at::Tensor& zeros,
                      int64_t group_size) {
    check_w4a16(x, packed_w, scales, zeros, group_size);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "w4a16_gemv expects x=[1,K]");
    int64_t K = x.size(1), N = packed_w.size(0);
    auto y = at::empty({1, N}, x.options());
    c10::cuda::CUDAGuard guard(x.device());
    w4a16_gemv_kernel<<<static_cast<unsigned>(N), 256, 0,
                        c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), packed_w.data_ptr<int32_t>(),
        scales.data_ptr<float>(), zeros.data_ptr<int32_t>(), reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        static_cast<int>(N), static_cast<int>(K), static_cast<int>(group_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor softmax(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat,
                "softmax CUDA backend expects CUDA FP32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 4 == 0,
                "softmax expects [M,N] with N divisible by 4");
    auto y = at::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    dim3 grid(static_cast<unsigned>(x.size(0)));
    softmax_v5_twopass<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "rms_norm expects CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
                "rms_norm expects FP32 tensors");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 1 && x.size(1) == weight.size(0) &&
                    x.size(1) % 4 == 0, "invalid RMSNorm shape");
    auto y = at::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    rmsnorm_v3_float4<<<static_cast<unsigned>(x.size(0)), 256, 0,
                        c10::cuda::getCurrentCUDAStream().stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor layer_norm(const at::Tensor& x, const at::Tensor& weight,
                      const at::Tensor& bias, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "layer_norm expects CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                    bias.scalar_type() == at::kFloat, "layer_norm expects FP32 tensors");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 1 && bias.dim() == 1 &&
                    x.size(1) == weight.size(0) && weight.sizes() == bias.sizes(),
                "invalid LayerNorm shape");
    auto y = at::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    layernorm_v2_warpshuffle<<<static_cast<unsigned>(x.size(0)), 256, 0,
                               c10::cuda::getCurrentCUDAStream().stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(),
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor batch_norm_infer(const at::Tensor& x, const at::Tensor& a,
                            const at::Tensor& b) {
    TORCH_CHECK(x.is_cuda() && a.is_cuda() && b.is_cuda(),
                "batch_norm_infer expects CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat && a.scalar_type() == at::kFloat &&
                    b.scalar_type() == at::kFloat, "batch_norm_infer expects FP32 tensors");
    TORCH_CHECK(x.dim() == 2 && a.dim() == 1 && b.dim() == 1 &&
                    x.size(1) == a.size(0) && a.sizes() == b.sizes(), "invalid BatchNorm shape");
    auto y = at::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    batchnorm_v2_rowblock<<<static_cast<unsigned>(x.size(0)), 256, 0,
                            c10::cuda::getCurrentCUDAStream().stream()>>>(
        x.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(),
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

#if defined(DANDAN_ENABLE_FLASHATTENTION)
template<bool CAUSAL>
at::Tensor flash_attention_impl(const at::Tensor& q, const at::Tensor& k,
                                const at::Tensor& v) {
    constexpr int D = 64, Br = 128, Bc = 64;
    const int64_t B = q.size(0), H = q.size(1), N = q.size(2);
    auto o = at::empty_like(q);
    const size_t smem = (static_cast<size_t>(4) * Bc * (D + 8) +
                         static_cast<size_t>(Br) * (D + 8)) * sizeof(__half);
    c10::cuda::CUDAGuard guard(q.device());
    cudaFuncSetAttribute(flashattention_v2_1<CAUSAL, Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    dim3 grid(static_cast<unsigned>((N + Br - 1) / Br), static_cast<unsigned>(B * H));
    flashattention_v2_1<CAUSAL, Br, Bc, D><<<grid, Br, smem,
                                              c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()), static_cast<int>(N));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
#endif

at::Tensor flash_attention(const at::Tensor& q, const at::Tensor& k,
                           const at::Tensor& v, bool causal) {
#if !defined(DANDAN_ENABLE_FLASHATTENTION)
    TORCH_CHECK(false, "FlashAttention CUDA kernel is not enabled in this Windows build");
#else
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "flash_attention expects CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                    v.scalar_type() == at::kHalf, "flash_attention expects FP16 tensors");
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
                    q.size(3) == 64, "flash_attention expects [B,H,N,64]");
    return causal ? flash_attention_impl<true>(q, k, v) : flash_attention_impl<false>(q, k, v);
#endif
}
} // namespace

TORCH_LIBRARY(dandan_ops, m) {
    m.def("w4a16_contract_version() -> int", []() -> int64_t { return 2; });
    m.def("gemm(Tensor a, Tensor b) -> Tensor");
    m.def("softmax(Tensor x) -> Tensor");
    m.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");
    m.def("layer_norm(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor");
    m.def("batch_norm_infer(Tensor x, Tensor a, Tensor b) -> Tensor");
    m.def("w4a16_gemm(Tensor x, Tensor packed_w, Tensor scales, Tensor zeros, int group_size) -> Tensor");
    m.def("w4a16_gemv(Tensor x, Tensor packed_w, Tensor scales, Tensor zeros, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(dandan_ops, CUDA, m) {
    m.impl("gemm", &gemm);
    m.impl("softmax", &softmax);
    m.impl("rms_norm", &rms_norm);
    m.impl("layer_norm", &layer_norm);
    m.impl("batch_norm_infer", &batch_norm_infer);
    m.impl("w4a16_gemm", &w4a16_gemm);
    m.impl("w4a16_gemv", &w4a16_gemv);
}

// The operators are registered through TORCH_LIBRARY and loaded with
// torch.ops.load_library. CUDAExtension still builds a Python module, so it
// needs the standard empty module entry point for the Windows linker.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
