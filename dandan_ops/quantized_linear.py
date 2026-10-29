"""Versioned INT4 state and inference dispatch (FP16 operands, FP32 sums)."""
import torch
from ._extension import load_extension
from ._dispatch import measured_route
from .quantization import validate_int4_state, dequantize_int4, pack_int4, quantize_int4

BACKENDS = ("auto", "cuda", "gemv", "small", "tensorcore", "naive", "reference")
FORMAT_VERSION = 2


def _check_bias(bias, n, device):
    if bias is not None and (not isinstance(bias, torch.Tensor) or bias.ndim != 1
            or bias.numel() != n or bias.device != device or not bias.is_floating_point()):
        raise ValueError("bias must be a floating tensor [N] on the input device")


def _select_backend(x, scales, group_size, backend, n=None):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    training = torch.is_grad_enabled() and (x.requires_grad or scales.requires_grad)
    supported = x.is_cuda and x.dtype == torch.float16 and group_size == 64 and not training
    reason = None
    if not supported:
        reason = "requires CUDA FP16, group_size=64 and forward inference"
    if backend == "reference":
        return "reference", "explicit reference requested"
    if not supported:
        if backend != "auto":
            raise ValueError(reason)
        return "reference", reason
    if backend == "gemv" and x.shape[0] != 1:
        raise ValueError("gemv requires M=1")
    if backend == "small" and x.shape[0] > 8:
        raise ValueError("small requires M<=8")
    if not load_extension(required=backend != "auto"):
        return "reference", "CUDA extension is unavailable"
    if backend == "naive":
        return "w4a16_gemm", None
    if backend == "small":
        return "w4a16_small", None
    # cuda still forces a custom implementation; the measured table contains
    # only packed kernels and never permits reference/full-weight fallback.
    if backend in ("auto", "cuda") and n is not None and x.is_contiguous() and x.data_ptr() % 16 == 0:
        route = measured_route("w4a16", x.shape[0], n, x.shape[1], x.device)
        if route:
            return route, None
    if backend == "gemv" or (backend in ("auto", "cuda") and x.shape[0] == 1):
        return "w4a16_gemv", None
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        if backend != "auto":
            raise ValueError("Tensor Core backend requires SM80 or newer")
        return "w4a16_gemm", "device lacks SM80 Tensor Core pipeline"
    return "w4a16_tensorcore", None


def quantized_linear(x, packed_weight, scales, zeros, *, in_features,
                     group_size=64, bias=None, backend="auto"):
    """y=cast(x @ W.T)+bias, where FP16 activations use FP16-rounded W.

    GEMV/MMA/naive/reference use the same weight rounding for FP16 inputs.
    Parallel reduction order can still differ; bitwise identity is not promised.
    BF16/FP32 inputs use FP32 dequant reference. Only auto allows fallback.
    """
    validate_int4_state(packed_weight, scales, zeros, in_features, group_size)
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != in_features:
        raise ValueError("x must have shape [M, in_features]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("x must be float16, bfloat16, or float32")
    if x.device != packed_weight.device:
        raise ValueError("x and quantized weight must share a device")
    _check_bias(bias, packed_weight.shape[0], x.device)
    route, _ = _select_backend(x, scales, group_size, backend, packed_weight.shape[0])
    if route == "reference":
        weight = dequantize_int4(packed_weight, scales, zeros, in_features, group_size)
        if x.dtype == torch.float16:
            weight = weight.half().float()
        y = (x.float() @ weight.t()).to(x.dtype)
    else:
        y = getattr(torch.ops.dandan_ops, route)(x.contiguous(), packed_weight.contiguous(),
                    scales.contiguous(), zeros.contiguous(), int(group_size))
    return y if bias is None else y + bias.to(y.dtype)


class QuantizedLinear(torch.nn.Module):
    """Packed INT4 state with versioned metadata and load-time validation.

    Value checks synchronize at construction/load only. Forward checks metadata
    only, so capture/replay never scans scales or copies them to CPU.
    """
    def __init__(self, packed_weight, scales, zeros, *, in_features,
                 group_size=64, bias=None, backend="auto"):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError("invalid quantized backend")
        validate_int4_state(packed_weight, scales, zeros, in_features, group_size, values=True)
        _check_bias(bias, packed_weight.shape[0], packed_weight.device)
        self.in_features, self.out_features = in_features, packed_weight.shape[0]
        self.group_size, self.backend = group_size, backend
        self.register_buffer("packed_weight", packed_weight.detach().contiguous())
        self.register_buffer("scales", scales.detach().contiguous())
        self.register_buffer("zeros", zeros.detach().contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().clone().contiguous())
        self.eval()

    def get_extra_state(self):
        return {"format_version": FORMAT_VERSION, "bits": 4, "scheme": "affine_unsigned",
                "packing": "int32_low_nibble_first", "original_shape": [self.out_features, self.in_features],
                "group_size": self.group_size, "fp16_weight_rounding": "fp16_before_multiply"}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if not isinstance(state, dict) or any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("incompatible INT4 checkpoint metadata/version/shape/group_size")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keys = [prefix + name for name in ("packed_weight", "scales", "zeros")]
        try:
            self.set_extra_state(state_dict.get(prefix + "_extra_state"))
            if all(key in state_dict for key in keys):
                validate_int4_state(*(state_dict[key] for key in keys), self.in_features,
                                    self.group_size, values=True)
                if state_dict[keys[0]].shape[0] != self.out_features:
                    raise ValueError("incompatible output dimension")
            bias = state_dict.get(prefix + "bias")
            if bias is not None:
                _check_bias(bias, self.out_features, state_dict[keys[0]].device)
        except (ValueError, TypeError, KeyError) as exc:
            error_msgs.append(f"{prefix}invalid INT4 state: {exc}")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_state_dict(cls, state, *, backend="auto"):
        meta = state.get("_extra_state")
        if not isinstance(meta, dict) or meta.get("format_version") != FORMAT_VERSION:
            raise ValueError("missing or unsupported INT4 format_version")
        shape = meta.get("original_shape")
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise ValueError("invalid original_shape metadata")
        layer = cls(state["packed_weight"], state["scales"], state["zeros"],
                    in_features=shape[1], group_size=meta["group_size"],
                    bias=state.get("bias"), backend=backend)
        layer.load_state_dict(state)
        return layer

    def _apply(self, fn, recurse=True):
        # Preserve the original FP32 scales, rather than upcasting a rounded copy.
        packed, scales, zeros = self.packed_weight, self.scales, self.zeros
        super()._apply(fn, recurse=recurse)
        device = self.packed_weight.device
        self.packed_weight = packed.to(device=device, dtype=torch.int32)
        self.scales = scales.to(device=device, dtype=torch.float32)
        self.zeros = zeros.to(device=device, dtype=torch.int32)
        return self

    @classmethod
    def from_float(cls, weight, bias=None, *, bits=4, group_size=64, backend="auto"):
        if not isinstance(bits, int) or isinstance(bits, bool) or bits != 4:
            raise ValueError("only bits=4 is supported")
        with torch.no_grad():
            q, scales, zeros = quantize_int4(weight, group_size)
        return cls(pack_int4(q), scales, zeros, in_features=weight.shape[1],
                   group_size=group_size, bias=bias, backend=backend)

    def execution_info(self, x):
        route, reason = _select_backend(x, self.scales, self.group_size, self.backend, self.out_features)
        return {"requested_backend": self.backend, "implementation": route,
                "fallback_reason": reason, "reason": reason or "selected packed inference backend",
                "input_copies": [i for i, t in enumerate((x, self.packed_weight, self.scales, self.zeros))
                                 if not t.is_contiguous()] if route != "reference" else [],
                "format_version": FORMAT_VERSION,
                "activation_dtype": str(x.dtype), "group_size": self.group_size}

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.packed_weight, self.scales, self.zeros))

    def dequantize(self):
        return dequantize_int4(self.packed_weight, self.scales, self.zeros, self.in_features, self.group_size)

    def forward(self, x):
        return quantized_linear(x, self.packed_weight, self.scales, self.zeros,
                                in_features=self.in_features, group_size=self.group_size,
                                bias=self.bias, backend=self.backend)
