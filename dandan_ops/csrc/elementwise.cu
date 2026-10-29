// Adapted from myvllm mini_engine/kernels/csrc/elementwise.cu.
// Source SHA256: 4938fc3c30b73cf79bca27d33a595e94685f16381de01d39b8ff2e9ebe3220c8
// Preserve Qwen3 FP16/BF16 rounding; no dependency on the engine checkout.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "elementwise_checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace {
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

inline void check_launch_capture_safe(cudaStream_t) { C10_CUDA_KERNEL_LAUNCH_CHECK(); }

__device__ float block_sum(float value, float* scratch) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < WARPS ? scratch[lane] : 0.f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            value += __shfl_down_sync(0xffffffff, value, offset);
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

template <typename scalar_t, bool RESIDUAL>
__global__ void norm_kernel(const scalar_t* x, const scalar_t* residual,
    const scalar_t* weight, scalar_t* output, scalar_t* summed, int64_t hidden, float eps) {
    const int64_t base = static_cast<int64_t>(blockIdx.x) * hidden;
    __shared__ float scratch[WARPS];
    float sum = 0.f;
    for (int64_t d = threadIdx.x; d < hidden; d += THREADS) {
        float value = static_cast<float>(x[base + d]);
        if (RESIDUAL) {
            // Eager residual_add returns a low-precision tensor before norm.
            scalar_t rounded = static_cast<scalar_t>(value + static_cast<float>(residual[base + d]));
            summed[base + d] = rounded;
            value = static_cast<float>(rounded);
        }
        sum += value * value;
    }
    const float inverse = rsqrtf(block_sum(sum, scratch) / static_cast<float>(hidden) + eps);
    for (int64_t d = threadIdx.x; d < hidden; d += THREADS) {
        float value = RESIDUAL ? static_cast<float>(summed[base + d]) : static_cast<float>(x[base + d]);
        // Qwen3 casts normalized activation before multiplying the weight.
        scalar_t normalized = static_cast<scalar_t>(value * inverse);
        output[base + d] = static_cast<scalar_t>(static_cast<float>(normalized) * static_cast<float>(weight[d]));
    }
}

template <typename scalar_t, bool SILU>
__global__ void pointwise_kernel(const scalar_t* x, const scalar_t* other,
                                 scalar_t* output, int64_t elements) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float value = static_cast<float>(x[index]);
        if (SILU) {
            // Match torch.silu(gate) * up: round SiLU before multiplication.
            scalar_t activated = static_cast<scalar_t>(value / (1.f + expf(-value)));
            output[index] = static_cast<scalar_t>(static_cast<float>(activated) * static_cast<float>(other[index]));
        } else {
            output[index] = static_cast<scalar_t>(value + static_cast<float>(other[index]));
        }
    }
}


template <typename scalar_t>
struct alignas(8) Vec4 {
    scalar_t value[4];
};

template <typename scalar_t, bool SILU>
__global__ void pointwise_vec4_kernel(const Vec4<scalar_t>* x, const Vec4<scalar_t>* other,
                                      Vec4<scalar_t>* output, int64_t vectors) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < vectors; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        Vec4<scalar_t> a = x[index], b = other[index], result;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float value = static_cast<float>(a.value[i]);
            if (SILU) {
                scalar_t activated = static_cast<scalar_t>(value / (1.f + expf(-value)));
                result.value[i] = static_cast<scalar_t>(static_cast<float>(activated) * static_cast<float>(b.value[i]));
            } else {
                result.value[i] = static_cast<scalar_t>(value + static_cast<float>(b.value[i]));
            }
        }
        output[index] = result;
    }
}

template <typename scalar_t, bool RESIDUAL>
void launch_norm(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight,
                  at::Tensor& output, at::Tensor& summed, float eps, cudaStream_t stream) {
    const int rows = static_cast<int>(x.numel() / x.size(-1));
    norm_kernel<scalar_t, RESIDUAL><<<rows, THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), RESIDUAL ? residual.data_ptr<scalar_t>() : nullptr,
        weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        RESIDUAL ? summed.data_ptr<scalar_t>() : nullptr, x.size(-1), eps);
    check_launch_capture_safe(stream);
}

template <typename scalar_t, bool SILU>
void launch_pointwise(const at::Tensor& x, const at::Tensor& other, at::Tensor& output, cudaStream_t stream) {
    const bool vectorized = x.numel() % 4 == 0
        && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0
        && reinterpret_cast<uintptr_t>(other.data_ptr()) % 8 == 0;
    const int64_t work = vectorized ? x.numel() / 4 : x.numel();
    const int blocks = static_cast<int>(std::min<int64_t>((work + THREADS - 1) / THREADS, 65535));
    if (vectorized) {
        pointwise_vec4_kernel<scalar_t, SILU><<<blocks, THREADS, 0, stream>>>(
            reinterpret_cast<const Vec4<scalar_t>*>(x.data_ptr<scalar_t>()),
            reinterpret_cast<const Vec4<scalar_t>*>(other.data_ptr<scalar_t>()),
            reinterpret_cast<Vec4<scalar_t>*>(output.data_ptr<scalar_t>()), work);
    } else {
        pointwise_kernel<scalar_t, SILU><<<blocks, THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), x.numel());
    }
    check_launch_capture_safe(stream);
}
} // namespace

at::Tensor rms_norm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps) {
    check_norm(x, weight, eps);
    check_lowp(x);
    const c10::cuda::CUDAGuard guard(x.device());
    auto output = at::empty_like(x);
    if (x.numel() == 0) return output;
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    at::Tensor unused;
    if (x.scalar_type() == at::kHalf)
        launch_norm<at::Half, false>(x, unused, weight, output, unused, static_cast<float>(eps), stream);
    else launch_norm<at::BFloat16, false>(x, unused, weight, output, unused, static_cast<float>(eps), stream);
    return output;
}

std::vector<at::Tensor> residual_rms_norm_cuda(const at::Tensor& x, const at::Tensor& residual,
                                              const at::Tensor& weight, double eps) {
    check_norm(x, weight, eps);
    check_elementwise(x, {residual});
    check_lowp(x);
    TORCH_CHECK(x.sizes() == residual.sizes(), "residual shape mismatch");
    const c10::cuda::CUDAGuard guard(x.device());
    auto output = at::empty_like(x);
    auto summed = at::empty_like(x);
    if (x.numel() == 0) return {output, summed};
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    if (x.scalar_type() == at::kHalf)
        launch_norm<at::Half, true>(x, residual, weight, output, summed, static_cast<float>(eps), stream);
    else launch_norm<at::BFloat16, true>(x, residual, weight, output, summed, static_cast<float>(eps), stream);
    return {output, summed};
}

at::Tensor silu_and_mul_cuda(const at::Tensor& gate, const at::Tensor& up) {
    check_elementwise(gate, {up});
    check_lowp(gate);
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate/up shape mismatch");
    const c10::cuda::CUDAGuard guard(gate.device());
    auto output = at::empty_like(gate);
    if (gate.numel() == 0) return output;
    auto stream = c10::cuda::getCurrentCUDAStream(gate.get_device()).stream();
    if (gate.scalar_type() == at::kHalf) launch_pointwise<at::Half, true>(gate, up, output, stream);
    else launch_pointwise<at::BFloat16, true>(gate, up, output, stream);
    return output;
}


TORCH_LIBRARY_FRAGMENT(dandan_ops, m) {
    m.def("elementwise_contract_version() -> int", []() -> int64_t { return 1; });
    m.def("rms_norm_lowp(Tensor x, Tensor weight, float eps) -> Tensor");
    m.def("residual_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps) -> Tensor[]");
    m.def("silu_and_mul(Tensor gate, Tensor up) -> Tensor");
}
TORCH_LIBRARY_IMPL(dandan_ops, CUDA, m) {
    m.impl("rms_norm_lowp", &rms_norm_cuda);
    m.impl("residual_rms_norm", &residual_rms_norm_cuda);
    m.impl("silu_and_mul", &silu_and_mul_cuda);
}
