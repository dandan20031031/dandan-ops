"""Small inference ops with explicit routing and PyTorch autograd fallbacks.

RMSNorm reduces in FP32 for FP16/BF16, rounds the normalized activation to
the input dtype, then multiplies by weight (the Qwen3 rounding convention).
Fused ops never mutate their inputs. CUDA Graph capture is supported.
"""
import math
import operator

import torch

from ._extension import load_extension


def _tensor(x, name):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _eps(value):
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > torch.finfo(torch.float32).max:
        raise ValueError("eps must be finite, nonnegative and representable as FP32")
    return value


def _validate(op, tensors, dim, eps):
    counts = {"softmax": 1, "rms_norm": 2, "layer_norm": 3,
              "batch_norm_infer": 3, "residual_rms_norm": 3, "silu_and_mul": 2}
    if op not in counts:
        raise ValueError(f"unknown elementwise operator: {op}")
    if len(tensors) != counts[op]:
        raise ValueError(f"{op} expects {counts[op]} tensors")
    for i, tensor in enumerate(tensors):
        _tensor(tensor, f"input {i}")
    x = tensors[0]
    if any(t.device != x.device for t in tensors):
        raise ValueError("inputs must share a device")
    if op == "softmax":
        dim = operator.index(dim)
        if not -max(1, x.ndim) <= dim < max(1, x.ndim):
            raise IndexError("softmax dim is out of range")
    elif op == "silu_and_mul":
        if tensors[1].shape != x.shape or tensors[1].dtype != x.dtype:
            raise ValueError("gate and up must share shape and dtype")
    else:
        if x.ndim < 1 or x.shape[-1] == 0:
            raise ValueError("normalization requires a non-empty last dimension")
        weights = tensors[1:]
        if op == "residual_rms_norm":
            residual = tensors[1]
            if residual.shape != x.shape or residual.dtype != x.dtype:
                raise ValueError("x and residual must share shape and dtype")
            weights = tensors[2:]
        if any(t.ndim != 1 or t.numel() != x.shape[-1] for t in weights):
            raise ValueError("normalization parameters must be [D], matching the last dimension")
        if op != "batch_norm_infer":
            eps = _eps(eps)
    return dim, eps


def elementwise_execution_info(op, *tensors, dim=-1, eps=1e-5, backend="auto"):
    """Explain routing and input copies without executing an operator.

    The six supported names match the public functions in this module.
    ``input_copies`` contains input indices copied for the CUDA path; it excludes
    ordinary result allocations and temporary tensors used by the torch fallback.
    """
    if backend not in ("auto", "cuda", "torch"):
        raise ValueError("backend must be auto, cuda, or torch")
    dim, eps = _validate(op, tensors, dim, eps)
    x = tensors[0]
    reason = None
    if not x.is_cuda:
        reason = "CUDA tensors required"
    elif torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        reason = "autograd requires torch"
    elif any(t.dtype != x.dtype for t in tensors):
        reason = "mixed dtypes require torch"
    elif x.numel() > 2**31 - 1 or any(d > 2**31 - 1 for d in x.shape):
        reason = "CUDA index range exceeded"
    elif op == "softmax" and (x.ndim != 2 or dim not in (-1, 1) or x.dtype != torch.float32):
        reason = "CUDA softmax requires FP32 [M,N] and the last dimension"
    elif op in ("layer_norm", "batch_norm_infer") and x.dtype != torch.float32:
        reason = "CUDA backend requires FP32"
    elif op == "rms_norm" and x.dtype == torch.float32 and x.shape[-1] % 4:
        reason = "FP32 CUDA RMSNorm requires D divisible by 4"
    elif op in ("residual_rms_norm", "silu_and_mul") and x.dtype not in (torch.float16, torch.bfloat16):
        reason = "fused CUDA backend requires FP16 or BF16"
    elif op == "rms_norm" and x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        reason = "CUDA RMSNorm requires FP16, BF16 or FP32"
    elif x.dtype == torch.bfloat16 and torch.cuda.get_device_capability(x.device)[0] < 8:
        reason = "BF16 CUDA backend requires SM80+"
    if backend == "torch":
        reason = "explicit torch requested"
    elif reason is None and not load_extension(required=backend == "cuda"):
        reason = "CUDA extension unavailable"
    if reason is not None and backend == "cuda":
        raise ValueError(reason)
    implementation = "torch" if reason is not None else "cuda"
    copies = []
    if implementation == "cuda":
        for i, t in enumerate(tensors):
            vector_alignment = op == "rms_norm" and x.dtype == torch.float32
            if not t.is_contiguous() or (vector_alignment and t.data_ptr() % 16):
                copies.append(i)
    return {"requested_backend": backend, "implementation": implementation,
            "reason": reason or "supported inference input", "input_copies": copies}


def _cuda_inputs(info, tensors):
    # contiguous() alone does not fix an already-contiguous offset view.
    return [
        (t.clone() if t.is_contiguous() else t.contiguous())
        if i in info["input_copies"] else t for i, t in enumerate(tensors)]


def _rms_reference(x, weight, eps):
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def softmax(x, dim=-1, *, backend="auto"):
    info = elementwise_execution_info("softmax", x, dim=dim, backend=backend)
    if info["implementation"] == "torch":
        return torch.softmax(x, dim)
    return torch.ops.dandan_ops.softmax(*_cuda_inputs(info, (x,)))


def rms_norm(x, weight, eps=1e-5, *, backend="auto"):
    info = elementwise_execution_info("rms_norm", x, weight, eps=eps, backend=backend)
    eps = _eps(eps)
    if info["implementation"] == "torch":
        return _rms_reference(x, weight, eps)
    a, w = _cuda_inputs(info, (x, weight))
    op = torch.ops.dandan_ops.rms_norm if x.dtype == torch.float32 else torch.ops.dandan_ops.rms_norm_lowp
    return op(a.view(-1, x.shape[-1]), w, eps).view(x.shape)


def layer_norm(x, weight, bias, eps=1e-5, *, backend="auto"):
    info = elementwise_execution_info("layer_norm", x, weight, bias, eps=eps, backend=backend)
    if info["implementation"] == "torch":
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, _eps(eps))
    a, w, b = _cuda_inputs(info, (x, weight, bias))
    return torch.ops.dandan_ops.layer_norm(a.view(-1, x.shape[-1]), w, b, _eps(eps)).view(x.shape)


def batch_norm_infer(x, a, b, *, backend="auto"):
    info = elementwise_execution_info("batch_norm_infer", x, a, b, backend=backend)
    if info["implementation"] == "torch":
        return x * a + b
    inputs, scale, bias = _cuda_inputs(info, (x, a, b))
    return torch.ops.dandan_ops.batch_norm_infer(inputs.view(-1, x.shape[-1]), scale, bias).view(x.shape)


def residual_rms_norm(x, residual, weight, eps=1e-5, *, backend="auto"):
    """Return (RMSNorm(summed), summed), rounding x+residual to x.dtype first."""
    info = elementwise_execution_info("residual_rms_norm", x, residual, weight, eps=eps, backend=backend)
    if info["implementation"] == "torch":
        summed = x + residual
        return _rms_reference(summed, weight, _eps(eps)), summed
    a, r, w = _cuda_inputs(info, (x, residual, weight))
    normalized, summed = torch.ops.dandan_ops.residual_rms_norm(
        a.view(-1, x.shape[-1]), r.view(-1, x.shape[-1]), w, _eps(eps))
    return normalized.view(x.shape), summed.view(x.shape)


def silu_and_mul(gate, up, *, backend="auto"):
    """SiLU(gate) * up, rounding SiLU to gate.dtype before multiplication."""
    info = elementwise_execution_info("silu_and_mul", gate, up, backend=backend)
    if info["implementation"] == "torch":
        return torch.nn.functional.silu(gate) * up
    return torch.ops.dandan_ops.silu_and_mul(*_cuda_inputs(info, (gate, up)))
