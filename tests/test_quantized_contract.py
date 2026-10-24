"""INT4 public contract, independent oracles and state lifecycle regressions."""
import copy
import io
from unittest.mock import Mock

import pytest
import torch
import dandan_ops as ops
from dandan_ops.quantization import validate_int4_state


@pytest.mark.parametrize("k", [1, 7, 8, 9, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("kind", ["zero", "constant", "positive", "negative", "tiny", "large"])
def test_quantization_edges(k, kind):
    torch.manual_seed(k)
    w = torch.randn(3, k)
    if kind == "zero": w.zero_()
    elif kind == "constant": w.fill_(2.5)
    elif kind == "positive": w.abs_()
    elif kind == "negative": w = -w.abs()
    elif kind == "tiny": w *= 1e-39
    elif kind == "large": w *= 1e30
    q, s, z = ops.quantize_int4(w)
    packed = ops.pack_int4(q)
    assert torch.equal(ops.unpack_int4(packed, k), q)
    assert torch.isfinite(s).all() and (s > 0).all()
    restored = ops.dequantize_int4(packed, s, z, k)
    assert torch.isfinite(restored).all()
    # Independently reconstruct by K->group indices, not the implementation loop.
    group = torch.arange(k) // 64
    expected = (q.float() - z[:, group].float()) * s[:, group]
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    if kind == "zero": assert torch.count_nonzero(restored) == 0
    elif kind != "tiny":
        assert ((restored.double() - w.double()).abs() <= s[:, group].double() * 1.01 + 1e-6).all()


def test_packed_word_bit_order_independent():
    q = torch.tensor([[0,1,2,3,4,5,6,15,9]], dtype=torch.uint8)
    words = ops.pack_int4(q)
    assert words.tolist() == [[0xF6543210 - 2**32, 9]]
    assert torch.equal(ops.unpack_int4(words, 9), q)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_weight(bad):
    with pytest.raises(ValueError, match="finite"):
        ops.quantize_int4(torch.tensor([[0., bad]]))


@pytest.mark.parametrize("group", [0, -1, True, None, 1.5])
def test_invalid_group(group):
    with pytest.raises(ValueError, match="group_size"):
        ops.QuantizedLinear.from_float(torch.ones(1, 8), group_size=group)


@pytest.mark.parametrize("field,value,pattern", [
    ("scales", 0., "positive"), ("scales", -1., "positive"),
    ("scales", float("inf"), "finite"), ("scales", float("nan"), "finite"),
    ("zeros", 16, "zero points"), ("zeros", -1, "zero points")])
def test_reject_bad_values_at_load(field, value, pattern):
    layer = ops.QuantizedLinear.from_float(torch.ones(3,65))
    state = copy.deepcopy(layer.state_dict())
    state[field].fill_(value)
    with pytest.raises((ValueError, RuntimeError), match=pattern):
        ops.QuantizedLinear.from_state_dict(state)
    # Destination must not be partially modified by an invalid checkpoint.
    before = layer.packed_weight.clone()
    with pytest.raises(RuntimeError, match="invalid INT4 state"):
        layer.load_state_dict(state)
    assert torch.equal(before, layer.packed_weight)


def test_checkpoint_roundtrip_dtype_and_no_source_alias():
    torch.manual_seed(14)
    w = torch.randn(7, 65, requires_grad=True)
    bias = torch.randn(7, requires_grad=True)
    layer = ops.QuantizedLinear.from_float(w, bias, backend="reference")
    assert not list(layer.parameters())
    assert not layer.scales.requires_grad and not layer.bias.requires_grad
    assert layer.bias.data_ptr() != bias.data_ptr()
    scales = layer.scales.clone()
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        layer.to(dtype=dtype)
        assert layer.scales.dtype == torch.float32
        assert torch.equal(layer.scales, scales)
        assert layer.packed_weight.dtype == layer.zeros.dtype == torch.int32
    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    clone = ops.QuantizedLinear.from_state_dict(state, backend="reference")
    x = torch.randn(3,65,dtype=torch.float16)
    torch.testing.assert_close(clone(x), layer(x), atol=0, rtol=0)
    # Includes non-tensor extra_state and works when nested in a parent model.
    parent = torch.nn.Sequential(layer)
    other = torch.nn.Sequential(clone)
    other.load_state_dict(parent.state_dict())
    torch.testing.assert_close(other(x), parent(x), atol=0, rtol=0)


@pytest.mark.parametrize("field,value", [("format_version",1),("packing","high_first"),
    ("group_size",128),("original_shape",[7,66]),("fp16_weight_rounding","fp32")])
def test_reject_incompatible_metadata(field, value):
    layer = ops.QuantizedLinear.from_float(torch.randn(7,65))
    state = copy.deepcopy(layer.state_dict())
    state["_extra_state"][field] = value
    # A destination knows its original K even when 65 and 66 share the same
    # packed/scale shapes. A factory must trust checkpoint original_shape.
    with pytest.raises((ValueError, RuntimeError)):
        layer.load_state_dict(state)
    if field != "original_shape":
        with pytest.raises((ValueError, RuntimeError)):
            ops.QuantizedLinear.from_state_dict(state)


def test_strict_backend_and_fallback_report():
    layer = ops.QuantizedLinear.from_float(torch.ones(3,65))
    x = torch.ones(1,65,dtype=torch.float16)
    report = layer.execution_info(x)
    assert report["implementation"] == "reference" and report["fallback_reason"]
    layer.backend = "cuda"
    with pytest.raises(ValueError, match="CUDA FP16"):
        layer(x)
    layer.backend = "reference"
    assert layer(x).shape == (1,3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_batch_graph_and_reference_share_rounding():
    import importlib
    mod = importlib.import_module("dandan_ops.quantized_linear")
    torch.manual_seed(51)
    layer = ops.QuantizedLinear.from_float(torch.randn(67,129,device="cuda",dtype=torch.float16))
    one = torch.randn(1,129,device="cuda",dtype=torch.float16)
    batch = torch.cat([one,torch.zeros_like(one),one * 2], dim=0)
    weight = layer.dequantize().half().float()
    expected = (one.float() @ weight.t()).half()
    for backend in ("cuda", "gemv", "tensorcore", "naive", "reference"):
        layer.backend = backend
        torch.testing.assert_close(layer(one), expected, atol=2e-3, rtol=2e-3)
    layer.backend = "cuda"
    eager = layer(batch)
    torch.testing.assert_close(eager[:1], layer(one), atol=2e-3, rtol=2e-3)
    graph = torch.cuda.CUDAGraph()
    # State is validated before capture. Forward must not scan Tensor values.
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*a,**kw): pytest.fail("forward synchronized or dequantized full weight")
        patch.setattr(mod, "dequantize_int4", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        for _ in range(3): layer(batch)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            output = layer(batch)
        for factor in (1., -1., 0.):
            batch.copy_(torch.cat([one,torch.zeros_like(one),one * 2],dim=0) * factor)
            graph.replay()
    # Last replay is the zero input, not a stale output captured once.
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    s = layer.scales.clone()
    layer.cpu().half().cuda().bfloat16()
    assert layer.scales.dtype == torch.float32 and torch.equal(layer.scales, s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k", [1,7,8,9,63,64,65,127,128,129])
def test_native_gemv_tails_and_empty_batch(k):
    torch.manual_seed(k)
    layer = ops.QuantizedLinear.from_float(torch.randn(5,k,device="cuda",dtype=torch.float16))
    x = torch.randn(1,k,device="cuda",dtype=torch.float16)
    expected = (x.float() @ layer.dequantize().half().float().t()).half()
    torch.testing.assert_close(layer(x), expected, atol=2e-3, rtol=2e-3)
    assert layer(x[:0]).shape == (0,5)


def test_invalid_functional_metadata():
    layer = ops.QuantizedLinear.from_float(torch.randn(3,65))
    x = torch.ones(2,65,dtype=torch.float16)
    with pytest.raises(TypeError, match="scales"):
        ops.quantized_linear(x,layer.packed_weight,layer.scales.half(),layer.zeros,in_features=65)
    with pytest.raises(ValueError, match="shape"):
        ops.quantized_linear(x,layer.packed_weight,layer.scales[:,:1],layer.zeros,in_features=65)
    with pytest.raises(ValueError, match="bias"):
        ops.quantized_linear(x,layer.packed_weight,layer.scales,layer.zeros,in_features=65,bias=torch.ones(4))
