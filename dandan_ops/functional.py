import math
import operator

import torch
from ._extension import load_extension
from ._dispatch import measured_route

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
        return {"requested_backend": backend, "implementation": route, "reason": reason}
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

def softmax(x, dim=-1):
    if dim not in (-1, x.ndim - 1):
        return torch.softmax(x, dim)
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda or x.shape[1] % 4:
        return torch.softmax(x, dim)
    if load_extension():
        return torch.ops.dandan_ops.softmax(x.contiguous())
    return torch.softmax(x, dim)

def rms_norm(x, weight, eps=1e-5):
    if x.ndim != 2 or weight.ndim != 1 or x.shape[-1] != weight.numel():
        raise ValueError("rms_norm expects x=[M,D] and weight=[D]")
    if x.is_cuda and x.dtype == weight.dtype == torch.float32 and x.shape[-1] % 4 == 0 and load_extension():
        return torch.ops.dandan_ops.rms_norm(x.contiguous(), weight.contiguous(), float(eps))
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight

def layer_norm(x, weight, bias, eps=1e-5):
    if (x.ndim != 2 or weight.ndim != 1 or bias.ndim != 1 or
            x.shape[-1] != weight.numel() or weight.numel() != bias.numel()):
        raise ValueError("layer_norm expects x=[M,D], weight/bias=[D]")
    if x.is_cuda and x.dtype == weight.dtype == bias.dtype == torch.float32 and load_extension():
        return torch.ops.dandan_ops.layer_norm(x.contiguous(), weight.contiguous(), bias.contiguous(), float(eps))
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)

def batch_norm_infer(x, a, b):
    if x.ndim != 2 or a.ndim != 1 or b.ndim != 1 or x.shape[-1] != a.numel() or a.numel() != b.numel():
        raise ValueError("batch_norm_infer expects x=[M,D] and a/b=[D]")
    if x.is_cuda and x.dtype == a.dtype == b.dtype == torch.float32 and load_extension():
        return torch.ops.dandan_ops.batch_norm_infer(x.contiguous(), a.contiguous(), b.contiguous())
    return x * a + b

def flash_attention(q, k, v, causal=False, scale=None, backend="auto", query_start=0):
    """Dense inference attention: Q=[B,Hq,Q,D], K/V=[B,Hkv,K,D].

    Query row i has absolute KV position query_start+i. Causal attention
    permits KV positions <= that position (True means visible in SDPA).
    Use query_start=0 for prefill and K-Q for a suffix/decode query.
    CUDA supports SM80+, FP16, D=128, GQA, and finite custom scales.
    auto falls back to SDPA for other dtypes/devices or autograd.
    Noncontiguous/unaligned inputs are copied for the CUDA backend.
    """
    if backend not in ("auto", "cuda", "torch"):
        raise ValueError("attention backend must be auto, cuda, or torch")
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in (q, k, v)):
        raise ValueError("q, k and v must be 4D tensors")
    if k.shape != v.shape:
        raise ValueError("q, k and v must be 4D with matching K/V shapes")
    if any(size == 0 for x in (q, k, v) for size in x.shape):
        raise ValueError("attention dimensions must be non-empty")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError("invalid GQA head dimensions")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise ValueError("attention tensors must share dtype and device")
    if not q.is_floating_point():
        raise ValueError("attention requires floating-point tensors")
    try:
        if isinstance(query_start, bool):
            raise TypeError
        query_start = operator.index(query_start)
    except TypeError as exc:
        raise ValueError("query_start must be an integer") from exc
    if query_start < 0 or query_start + q.shape[2] > k.shape[2]:
        raise ValueError("query_start must map Q rows into K rows (0 <= start <= K-Q)")
    if scale is not None:
        scale = float(scale)
        if not math.isfinite(scale) or abs(scale) > torch.finfo(torch.float32).max:
            raise ValueError("scale must be finite and representable as FP32")
    training = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))
    supported = (q.is_cuda and q.dtype == torch.float16 and q.shape[-1] == 128
                 and not training and torch.cuda.get_device_capability(q.device)[0] >= 8)
    if backend != "torch" and supported and load_extension(required=backend == "cuda"):
        def aligned(x):
            x = x.contiguous()
            return x if x.data_ptr() % 16 == 0 else x.clone()
        return torch.ops.dandan_ops.flash_inference(
            aligned(q), aligned(k), aligned(v), query_start, bool(causal),
            q.shape[-1] ** -0.5 if scale is None else scale)
    if backend == "cuda":
        raise ValueError("CUDA flash_attention requires SM80+, FP16, head_dim=128, and forward inference")
    mask = None
    if causal and query_start:
        rows = torch.arange(q.shape[2], device=q.device)[:, None] + query_start
        cols = torch.arange(k.shape[2], device=q.device)[None, :]
        mask = cols <= rows
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=bool(causal) and mask is None,
        scale=scale, enable_gqa=True)
