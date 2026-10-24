"""Optional myvllm adapter without modifying the engine checkout.

Call after loading/moving a floating NativeQwen3 and before allocating graph
buckets. The original engine reference quantizer is used once; no re-quantizing
already packed values. This module imports myvllm only when explicitly used.
"""
import torch
from torch import nn

from ..quantized_linear import QuantizedLinear
from ..quantization import pack_int4
from .._extension import load_extension


def convert_signed_int4(weight, scales, in_features, source_group_size=128):
    """Lossless conversion of uint8 two's-complement INT4 into affine INT4.

    Source logical q in [-8,7], low nibble first; storage includes padded K.
    Target q=q_signed+8, zero=8; one source group becomes two 64-wide groups.
    Validated values once here; never called during forward/graph replay.
    """
    if source_group_size != 128:
        raise ValueError("myvllm bridge requires source_group_size=128")
    if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
        raise ValueError("in_features must be a positive integer")
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8:
        raise TypeError("source packed weight must be uint8")
    if weight.ndim != 2 or min(weight.shape) <= 0 or weight.device.type not in ("cpu", "cuda"):
        raise ValueError("source packed weight must be non-empty [N, padded_K/2] on CPU/CUDA")
    if not isinstance(scales, torch.Tensor) or scales.dtype != torch.float32:
        raise TypeError("source scales must be FP32")
    groups = (in_features + 127) // 128
    if weight.shape[1] != groups * 64 or scales.shape != (weight.shape[0], groups):
        raise ValueError("invalid source packed weight or scales shape")
    if weight.device != scales.device:
        raise ValueError("source packed weight and scales must share a device")
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError("source scales must be finite and strictly positive")
    with torch.no_grad():
        # Two's-complement nibble XOR 8 equals signed_q+8 exactly.
        nibbles = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(1)[:, :in_features]
        words = pack_int4(nibbles ^ 8)
        target_scales = scales.repeat_interleave(2, dim=1)[:, :(in_features + 63) // 64].contiguous()
        zeros = torch.full_like(target_scales, 8, dtype=torch.int32)
    return words, target_scales, zeros


def _validate_config(config):
    if (config.canonical_weight_dtype != "int4" or config.activation_dtype != "fp16"
            or config.group_size != 128 or not config.symmetric):
        raise ValueError("myvllm dandan adapter requires symmetric INT4, FP16 activation, group_size=128")


class EngineQuantLinear(nn.Module):
    """NativeQwen3 projection adapter, retaining arbitrary leading dimensions."""
    def __init__(self, linear, config):
        super().__init__()
        _validate_config(config)
        if not isinstance(linear, nn.Linear):
            raise TypeError("source projection must be nn.Linear")
        if linear.weight.device.type != "cuda" or linear.weight.dtype != torch.float16:
            raise ValueError("source model must be CUDA FP16")
        load_extension(required=True)
        from mini_engine.quantization.backends import Int4WeightOnlyBackend
        source = Int4WeightOnlyBackend(128).pack(linear.weight)
        words, scales, zeros = convert_signed_int4(source.weight, source.scale, linear.in_features)
        self.op = QuantizedLinear(words, scales, zeros, in_features=linear.in_features,
                                  group_size=64, bias=linear.bias, backend="cuda")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.eval()

    @property
    def storage_bytes(self):
        return self.op.storage_bytes

    def dequantize(self):
        return self.op.dequantize()

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("input feature dimension mismatch")
        # Strict CUDA backend intentionally raises on BF16/CPU/training.
        y = self.op(x.reshape(-1, self.in_features))
        return y.reshape(*x.shape[:-1], self.out_features)


def install_int4_projections(model, config):
    """Opt-in replacement, all-or-nothing preparation before model mutation.

    Keeps embedding/norm/lm_head, tied parameters and KV/cache APIs unchanged.
    Projection fusion is deliberately rejected until packed fusion is validated.
    """
    _validate_config(config)
    if getattr(model, "quantization_report", None) is not None:
        raise ValueError("model is already quantized; reload floating weights")
    if getattr(model, "projection_fusion_enabled", False):
        raise ValueError("projection fusion and dandan quantization are not combined yet")
    selected = [(name, mod) for name, mod in model.named_modules()
                if name.startswith("model.layers.") and isinstance(mod, nn.Linear)]
    if not selected:
        raise ValueError("no NativeQwen3 decoder projections found")
    replacements = [(name, EngineQuantLinear(mod, config)) for name, mod in selected]
    original_bytes = sum(mod.weight.numel() * mod.weight.element_size() for _, mod in selected)
    packed_bytes = sum(mod.storage_bytes for _, mod in replacements)
    for name, replacement in replacements:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, replacement)
    report = {
        "weight_format": "int4", "activation_dtype": "fp16", "format_version": 2,
        "compute_backend": "dandan_w4a16_small_gemv_or_tensorcore", "reference_only": False,
        "accelerated_quant_gemm": True, "source_group_size": 128, "storage_group_size": 64,
        "replaced_layers": [name for name, _ in replacements],
        "original_projection_weight_bytes": original_bytes,
        "packed_projection_weight_and_scale_bytes": packed_bytes,
        "projection_storage_ratio": original_bytes / packed_bytes,
        "high_precision_modules": ["embedding", "all_norms", "residual", "lm_head", "kv_cache"],
    }
    model.quantization_report = report
    return report
