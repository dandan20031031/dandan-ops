#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "../../kernels/hgemm_tma.cuh"

namespace {
void tensor_map(CUtensorMap* map, const half* ptr, int rows, int cols, int box_rows) {
    cuuint64_t dims[2] = {cuuint64_t(cols), cuuint64_t(rows)};
    cuuint64_t strides[1] = {cuuint64_t(cols) * 2};
    cuuint32_t box[2] = {64, cuuint32_t(box_rows)}, element[2] = {1, 1};
    const auto result = cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
        const_cast<half*>(ptr), dims, strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(result));
}

at::Tensor hgemm_tma(const at::Tensor& a, const at::Tensor& bt) {
    TORCH_CHECK(a.is_cuda() && bt.is_cuda() && a.device() == bt.device(),
                "TMA expects tensors on the same CUDA device");
    TORCH_CHECK(a.dim() == 2 && bt.dim() == 2 && a.size(1) == bt.size(1),
                "TMA expects A=[M,K], weight=[N,K]");
    TORCH_CHECK(a.scalar_type() == at::kHalf && bt.scalar_type() == at::kHalf &&
                a.is_contiguous() && bt.is_contiguous(), "TMA expects contiguous FP16 tensors");
    const int64_t M = a.size(0), N = bt.size(0), K = a.size(1);
    TORCH_CHECK(M > 0 && M % 128 == 0 && N > 0 && N % 64 == 0 && K > 0 && K % 64 == 0 &&
                a.numel() <= INT_MAX && bt.numel() <= INT_MAX && M * N <= INT_MAX,
                "TMA requires M%128=0, N%64=0, K%64=0 and 32-bit index range");
    TORCH_CHECK((uintptr_t(a.const_data_ptr()) & 15) == 0 && (uintptr_t(bt.const_data_ptr()) & 15) == 0,
                "TMA requires 16-byte aligned inputs");
    c10::cuda::CUDAGuard guard(a.device());
    int major;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, a.get_device()));
    TORCH_CHECK(major == 12,
                "TMA backend currently validated only on SM120");
    alignas(64) CUtensorMap map_a, map_b;
    tensor_map(&map_a, reinterpret_cast<const half*>(a.data_ptr<at::Half>()), int(M), int(K), 128);
    tensor_map(&map_b, reinterpret_cast<const half*>(bt.data_ptr<at::Half>()), int(N), int(K), 64);
    auto out = at::empty({M, N}, a.options());
    auto kernel = dandan_tma::hgemm_v10_tma_kernel<128, 64, 64, 8, 1, 2, 16, 2>;
    constexpr int shared = 2 * (128 + 64) * 64 * 2 + 1024 + 32;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    const int blocks = ((int(M) / 128 + 15) / 16) * ((int(N) / 64 + 1) / 2) * 32;
    kernel<<<blocks, 256, shared, c10::cuda::getCurrentCUDAStream().stream()>>>(
        map_a, map_b, reinterpret_cast<half*>(out.data_ptr<at::Half>()), int(M), int(N), int(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}
TORCH_LIBRARY_FRAGMENT(dandan_ops, m) { m.def("hgemm_tma(Tensor a, Tensor weight) -> Tensor"); }
TORCH_LIBRARY_IMPL(dandan_ops, CUDA, m) { m.impl("hgemm_tma", &hgemm_tma); }
