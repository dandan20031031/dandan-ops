#pragma once
#include <ATen/ATen.h>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <initializer_list>

inline void check_elementwise(const at::Tensor& x,
                              std::initializer_list<at::Tensor> others = {}) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "expected contiguous CUDA tensors");
    TORCH_CHECK(x.numel() <= INT_MAX, "CUDA elementwise index range exceeded");
    TORCH_CHECK(!c10::GradMode::is_enabled() || !x.requires_grad(),
                "CUDA elementwise operators are inference-only");
    for (const auto& t : others) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                    "inputs must share dtype and device");
        TORCH_CHECK(t.is_contiguous(), "expected contiguous CUDA tensors");
        TORCH_CHECK(!c10::GradMode::is_enabled() || !t.requires_grad(),
                    "CUDA elementwise operators are inference-only");
    }
}

inline void check_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
    check_elementwise(x, {weight});
    TORCH_CHECK(x.dim() == 2 && x.size(1) > 0 && x.size(1) <= INT_MAX &&
                weight.dim() == 1 && weight.numel() == x.size(1),
                "normalization expects [M,D], weight=[D], D>0");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0 && eps <= static_cast<double>(FLT_MAX),
                "eps must be finite, nonnegative and representable as FP32");
}

inline void check_lowp(const at::Tensor& x) {
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "CUDA fused operators require FP16 or BF16");
}
