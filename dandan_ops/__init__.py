"""Unified Python API for the final CUDA operators in this repository."""

from .functional import (
    gemm,
    gemm_execution_info,
    softmax,
    rms_norm,
    layer_norm,
    batch_norm_infer,
    flash_attention,
    flash_attention_execution_info,
    residual_rms_norm,
    silu_and_mul,
    elementwise_execution_info,
)
from .quantization import quantize_int4, dequantize_int4, pack_int4, unpack_int4
from .quantized_linear import QuantizedLinear, quantized_linear
from .attention import (flash_attention_varlen, paged_attention,
                        flash_attention_varlen_execution_info, paged_attention_execution_info,
                        prepare_varlen_attention, prepare_paged_attention,
                        VarlenAttentionPlan, PagedAttentionPlan)

__all__ = [
    "gemm",
    "gemm_execution_info",
    "softmax",
    "rms_norm",
    "layer_norm",
    "batch_norm_infer",
    "flash_attention",
    "flash_attention_execution_info",
    "flash_attention_varlen",
    "paged_attention",
    "flash_attention_varlen_execution_info",
    "paged_attention_execution_info",
    "prepare_varlen_attention",
    "prepare_paged_attention",
    "VarlenAttentionPlan",
    "PagedAttentionPlan",
    "residual_rms_norm",
    "silu_and_mul",
    "elementwise_execution_info",
    "quantize_int4",
    "dequantize_int4",
    "pack_int4",
    "unpack_int4",
    "QuantizedLinear",
    "quantized_linear",
]
