#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "gemm_checks.h"
#include "../../kernels/hgemm_pipeline.cuh"
#include "../../kernels/w4a16_small.cuh"

namespace {
void check_hgemm(const at::Tensor& a, const at::Tensor& bt) {
    TORCH_CHECK(a.is_cuda() && bt.is_cuda() && a.device() == bt.device(),
                "hgemm expects tensors on the same CUDA device");
    TORCH_CHECK(a.dim() == 2 && bt.dim() == 2 && a.size(1) == bt.size(1),
                "hgemm expects A=[M,K], weight=[N,K]");
    TORCH_CHECK(a.scalar_type() == at::kHalf && bt.scalar_type() == at::kHalf,
                "hgemm expects FP16 tensors");
    TORCH_CHECK(a.is_contiguous() && bt.is_contiguous(), "hgemm expects contiguous tensors");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && bt.size(0) > 0 &&
                a.size(0) <= INT_MAX - 128 && a.size(1) <= INT_MAX - 64 && bt.size(0) <= INT_MAX - 64 &&
                a.numel() <= INT_MAX && bt.numel() <= INT_MAX && a.size(0) * bt.size(0) <= INT_MAX,
                "invalid hgemm dimensions");
}

template<bool Q, int BM>
void launch(const at::Tensor& a, const at::Tensor& b, const float* scales,
            const int32_t* zeros, at::Tensor& c) {
    constexpr int threads = BM / 16 * (BM <= 32 ? 4 : 1) * 32;
    constexpr int shared = 2 * (BM + 64) * 64 * sizeof(half);
    const int M = int(a.size(0)), K = int(a.size(1)), N = int(b.size(0));
    auto kernel = dandan_hgemm::pipeline_kernel<Q, BM>;
    if (shared > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    kernel<<<((M + BM - 1) / BM) * ((N + 63) / 64), threads, shared,
              c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()), b.const_data_ptr(), scales, zeros,
        reinterpret_cast<half*>(c.data_ptr<at::Half>()), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool Q>
at::Tensor run_pipeline(const at::Tensor& a, const at::Tensor& b,
                        const float* scales = nullptr, const int32_t* zeros = nullptr) {
    c10::cuda::CUDAGuard guard(a.device());
    int major;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, a.get_device()));
    TORCH_CHECK(major >= 8,
                "hgemm pipeline requires SM80 or newer");
    auto c = at::empty({a.size(0), b.size(0)}, a.options());
    if (a.size(0) == 0) return c;
    if (a.size(0) <= 16) launch<Q, 16>(a, b, scales, zeros, c);
    else if (a.size(0) <= 32) launch<Q, 32>(a, b, scales, zeros, c);
    else launch<Q, 128>(a, b, scales, zeros, c);
    return c;
}

at::Tensor hgemm(const at::Tensor& a, const at::Tensor& bt) {
    check_hgemm(a, bt);
    return run_pipeline<false>(a, bt);
}

at::Tensor w4a16_tensorcore(const at::Tensor& x, const at::Tensor& packed,
                           const at::Tensor& scales, const at::Tensor& zeros, int64_t group_size) {
    check_w4a16(x, packed, scales, zeros, group_size);
    TORCH_CHECK(group_size == 64, "W4A16 Tensor Core backend requires group_size=64");
    return run_pipeline<true>(x, packed, scales.data_ptr<float>(), zeros.data_ptr<int32_t>());
}

at::Tensor w4a16_small(const at::Tensor& x, const at::Tensor& packed,
                      const at::Tensor& scales, const at::Tensor& zeros, int64_t group_size) {
    check_w4a16(x, packed, scales, zeros, group_size);
    TORCH_CHECK(group_size == 64 && x.size(0) <= 8, "W4A16 small requires group_size=64 and M<=8");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({x.size(0), packed.size(0)}, x.options());
    if (x.size(0) == 0) return out;
    auto kernel = x.size(0) == 1 ? dandan_w4::small_batch<1> :
        x.size(0) <= 2 ? dandan_w4::small_batch<2> :
        x.size(0) <= 4 ? dandan_w4::small_batch<4> : dandan_w4::small_batch<8>;
    kernel<<<(packed.size(0) + 3) / 4, 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), packed.data_ptr<int32_t>(),
        scales.data_ptr<float>(), zeros.data_ptr<int32_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), int(x.size(0)), int(packed.size(0)), int(x.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(dandan_ops, m) {
    m.def("hgemm(Tensor a, Tensor weight) -> Tensor");
    m.def("w4a16_tensorcore(Tensor x, Tensor packed_w, Tensor scales, Tensor zeros, int group_size) -> Tensor");
    m.def("w4a16_small(Tensor x, Tensor packed_w, Tensor scales, Tensor zeros, int group_size) -> Tensor");
}
TORCH_LIBRARY_IMPL(dandan_ops, CUDA, m) {
    m.impl("hgemm", &hgemm);
    m.impl("w4a16_tensorcore", &w4a16_tensorcore);
    m.impl("w4a16_small", &w4a16_small);
}
