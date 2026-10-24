#pragma once
#include <ATen/ATen.h>
#include <climits>
#include <cstdint>

inline void check_w4a16(const at::Tensor& x, const at::Tensor& packed,
                       const at::Tensor& scales, const at::Tensor& zeros,
                       int64_t group_size) {
    TORCH_CHECK(x.dim() == 2 && packed.dim() == 2 && scales.dim() == 2 && zeros.dim() == 2,
                "W4A16 tensors must be 2D");
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
                "W4A16 expects CUDA tensors");
    TORCH_CHECK(x.device() == packed.device() && x.device() == scales.device() && x.device() == zeros.device(),
                "W4A16 tensors must share a device");
    TORCH_CHECK(x.scalar_type() == at::kHalf && packed.scalar_type() == at::kInt &&
                scales.scalar_type() == at::kFloat && zeros.scalar_type() == at::kInt,
                "invalid W4A16 dtypes");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && scales.is_contiguous() && zeros.is_contiguous(),
                "W4A16 tensors must be contiguous");
    const auto M = x.size(0), K = x.size(1), N = packed.size(0);
    TORCH_CHECK(K > 0 && N > 0 && K <= INT_MAX - 64 && N <= INT_MAX - 64 && M <= INT_MAX - 128,
                "invalid W4A16 dimensions");
    TORCH_CHECK(group_size > 0 && group_size <= INT_MAX - K, "invalid group_size");
    TORCH_CHECK(x.numel() <= INT_MAX && M * N <= INT_MAX && N * K <= INT_MAX,
                "W4A16 dimensions exceed the kernel index range");
    TORCH_CHECK(packed.size(1) == (K + 7) / 8, "invalid packed weight shape");
    const auto groups = (K + group_size - 1) / group_size;
    TORCH_CHECK(scales.sizes() == at::IntArrayRef({N, groups}) && zeros.sizes() == scales.sizes(),
                "invalid W4A16 metadata shape");
}
