"""W4A16 quantization utilities used by QuantizedLinear.

The storage format is row-major [out_features, in_features]: eight unsigned
INT4 values are packed into one int32 word. Scales and zero points are stored
per output row and per K group.
"""
import torch
import torch.nn.functional as F


def _check_group_size(group_size):
    if (not isinstance(group_size, int) or isinstance(group_size, bool)
            or group_size <= 0):
        raise ValueError("group_size must be a positive integer")


def _check_weight(weight):
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a Tensor")
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("weight must be a non-empty [N, K] tensor")
    if weight.dtype not in (torch.float16, torch.float32):
        raise TypeError("weight must be float16 or float32")
    if weight.device.type not in ("cpu", "cuda"):
        raise ValueError("weight must be on CPU or CUDA")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must contain only finite values")


def validate_int4_state(packed, scales, zeros, in_features, group_size=64, *, values=False):
    """Validate metadata without synchronizing; values=True is load-time only.

    Tensor values are scanned at construction/checkpoint loading, never inside
    the token hot path. Direct low-level callers own the value contract.
    """
    _check_group_size(group_size)
    if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
        raise ValueError("in_features must be a positive integer")
    for tensor, name, dtype in ((packed, "packed_weight", torch.int32),
                               (scales, "scales", torch.float32), (zeros, "zeros", torch.int32)):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise TypeError(f"{name} must be a {dtype} Tensor")
        if tensor.ndim != 2 or min(tensor.shape) <= 0:
            raise ValueError(f"{name} must be a non-empty 2D tensor")
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must be on CPU or CUDA")
    n = packed.shape[0]
    if packed.shape[1] != (in_features + 7) // 8:
        raise ValueError("invalid packed weight shape")
    expected = (n, (in_features + group_size - 1) // group_size)
    if scales.shape != expected or zeros.shape != expected:
        raise ValueError("invalid scales or zeros shape")
    if packed.device != scales.device or packed.device != zeros.device:
        raise ValueError("quantization tensors must share a device")
    if values:
        if not bool((torch.isfinite(scales) & (scales > 0)).all()):
            raise ValueError("scales must be finite and strictly positive")
        if not bool(((zeros >= 0) & (zeros <= 15)).all()):
            raise ValueError("zero points must be in [0,15]")


def quantize_int4(weight, group_size=64):
    """Quantize a floating-point [N, K] weight with asymmetric affine INT4."""
    _check_weight(weight)
    _check_group_size(group_size)
    n, k = weight.shape
    groups = (k + group_size - 1) // group_size
    padded_k = groups * group_size
    w = weight.float()
    if padded_k != k:
        w = F.pad(w, (0, padded_k - k))
    blocks = w.reshape(n, groups, group_size)
    lower = blocks.amin(dim=-1).clamp(max=0)
    upper = blocks.amax(dim=-1).clamp(min=0)
    # Divide endpoints first: upper-lower may overflow even for finite FP32.
    raw_scale = upper / 15.0 - lower / 15.0
    scale = torch.where((upper == 0) & (lower == 0), torch.ones_like(raw_scale),
                        raw_scale.clamp_min(torch.finfo(torch.float32).tiny))
    zero = torch.round(-lower / scale).clamp(0, 15).to(torch.int32)
    q = torch.round(blocks / scale.unsqueeze(-1) + zero.unsqueeze(-1))
    q = q.clamp(0, 15).to(torch.uint8).reshape(n, padded_k)
    return q[:, :k].contiguous(), scale.float().contiguous(), zero.contiguous()


def pack_int4(qweight):
    """Pack [N, K] uint8 values in [0, 15] into [N, ceil(K/8)] int32."""
    if not isinstance(qweight, torch.Tensor):
        raise TypeError("qweight must be a Tensor")
    if qweight.ndim != 2 or min(qweight.shape) <= 0:
        raise ValueError("qweight must be a non-empty 2D tensor")
    if qweight.dtype != torch.uint8:
        raise TypeError("qweight must be uint8")
    if torch.any(qweight > 15):
        raise ValueError("qweight values must be in [0, 15]")
    n, k = qweight.shape
    packed_k = (k + 7) // 8
    padded_k = packed_k * 8
    if padded_k != k:
        qweight = F.pad(qweight, (0, padded_k - k))
    packed = torch.zeros((n, packed_k), dtype=torch.int64, device=qweight.device)
    for nibble in range(8):
        packed |= qweight[:, nibble::8].to(torch.int64) << (4 * nibble)
    return packed.to(torch.int32)


def unpack_int4(packed, original_k=None):
    """Unpack int32 words into uint8 INT4 values."""
    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed must be a Tensor")
    if packed.ndim != 2 or min(packed.shape) <= 0:
        raise ValueError("packed must be a non-empty 2D tensor")
    if packed.dtype != torch.int32:
        raise TypeError("packed must be int32")
    values = [((packed >> (4 * nibble)) & 0xF) for nibble in range(8)]
    q = torch.stack(values, dim=-1).reshape(packed.shape[0], -1).to(torch.uint8)
    if original_k is None:
        return q
    if (not isinstance(original_k, int) or isinstance(original_k, bool)
            or original_k <= 0 or (original_k + 7) // 8 != packed.shape[1]):
        raise ValueError("invalid original_k")
    return q[:, :original_k]


def dequantize_int4(packed, scales, zeros, original_k, group_size=64):
    """Dequantize packed INT4 storage into an FP32 [N, K] tensor."""
    validate_int4_state(packed, scales, zeros, original_k, group_size)
    q = unpack_int4(packed, original_k)
    if scales.dtype != torch.float32 or zeros.dtype != torch.int32:
        raise TypeError("scales must be float32 and zeros must be int32")
    groups = (original_k + group_size - 1) // group_size
    expected = (q.shape[0], groups)
    if tuple(scales.shape) != expected or tuple(zeros.shape) != expected:
        raise ValueError("invalid scales or zeros shape")
    if scales.device != q.device or zeros.device != q.device:
        raise ValueError("all quantization tensors must be on the same device")
    out = torch.empty((q.shape[0], original_k), dtype=torch.float32, device=q.device)
    for group in range(groups):
        start = group * group_size
        end = min(start + group_size, original_k)
        out[:, start:end] = (q[:, start:end].float()
                             - zeros[:, group].float().unsqueeze(1)) \
                            * scales[:, group].unsqueeze(1)
    return out
