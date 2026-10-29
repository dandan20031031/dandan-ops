import torch
from ._extension import load_extension
from ._dispatch import measured_route
from .attention import flash_attention, flash_attention_execution_info
from .elementwise import (softmax, rms_norm, layer_norm, batch_norm_infer,
                          residual_rms_norm, silu_and_mul, elementwise_execution_info)

def _matrix(x, name):
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if x.numel() == 0:
        raise ValueError(f"{name} must be non-empty")

def gemm_execution_info(a, b, *, weight_transposed=False, backend="auto"):
    """Return the selected implementation and reason without executing GEMM."""
    _matrix(a, "a"); _matrix(b, "b")
    if backend not in ("auto", "cuda", "tma", "torch"):
        raise ValueError("backend must be auto, cuda, tma, or torch")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("GEMM inputs must have the same dtype and device")
    if a.shape[1] != b.shape[1 if weight_transposed else 0]:
        raise ValueError("incompatible GEMM dimensions")
    training = torch.is_grad_enabled() and (a.requires_grad or b.requires_grad)
    supported = (a.is_cuda and a.dtype == torch.float16 and not training
                 and torch.cuda.get_device_capability(a.device)[0] >= 8)
    def info(route, reason):
        inputs = (a, b if weight_transposed else b.t())
        copies = [i for i, t in enumerate(inputs) if not t.is_contiguous()] if route != "torch" else []
        return {"requested_backend": backend, "implementation": route, "reason": reason,
                "input_copies": copies}
    if backend == "torch":
        return info("torch", "explicit torch requested")
    if backend == "auto" and not supported:
        return info("torch", "device/dtype/autograd requires torch")
    if not supported:
        raise ValueError("custom HGEMM requires CUDA SM80+, FP16 inputs, and forward inference")
    if backend == "auto" and weight_transposed and a.is_contiguous() and b.is_contiguous():
        route = measured_route("dense", a.shape[0], b.shape[0], a.shape[1], a.device)
        if route and (route != "tma" or (a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0)):
            if route == "torch" or load_extension():
                return info(route, "offline measured profile")
    if not load_extension(required=backend != "auto"):
        return info("torch", "CUDA extension unavailable")
    bt = b if weight_transposed else b.t()
    use_tma = backend == "tma" or (backend == "auto"
        and torch.cuda.get_device_capability(a.device) == (12, 0)
        and a.shape[0] % 128 == 0 and bt.shape[0] % 64 == 0 and a.shape[1] % 64 == 0
        and a.data_ptr() % 16 == 0 and bt.data_ptr() % 16 == 0)
    return info("tma" if use_tma else "cuda", "explicit request" if backend != "auto" else "unprofiled supported-shape default")


def gemm(a, b, *, weight_transposed=False, backend="auto"):
    """FP16 GEMM, a=[M,K], b=[K,N] or weight_transposed=True b=[N,K].

    auto uses exact-shape offline measurements in their matching environment,
    then the original support-based default. cuda/tma/torch force a backend.
    """
    route = gemm_execution_info(a, b, weight_transposed=weight_transposed, backend=backend)["implementation"]
    if route == "torch":
        return a @ (b.t() if weight_transposed else b)
    a = a.contiguous()
    bt = (b if weight_transposed else b.t()).contiguous()
    op = torch.ops.dandan_ops.hgemm_tma if route == "tma" else torch.ops.dandan_ops.hgemm
    return op(a, bt)
