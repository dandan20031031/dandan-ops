import torch
import dandan_ops

def test_cpu_fallbacks():
    x = torch.randn(3, 8)
    w = torch.randn(8)
    b = torch.randn(8)
    other = torch.randn(8, 4)
    torch.testing.assert_close(dandan_ops.gemm(x, other), x @ other)
    torch.testing.assert_close(dandan_ops.softmax(x), torch.softmax(x, -1))
    torch.testing.assert_close(dandan_ops.rms_norm(x, w), x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w)
    torch.testing.assert_close(dandan_ops.layer_norm(x, w, b), torch.nn.functional.layer_norm(x, (8,), w, b, 1e-5))
    torch.testing.assert_close(dandan_ops.batch_norm_infer(x, w, b), x * w + b)


def test_int4_pack_roundtrip_partial_k():
    weight = torch.randn(3, 65, dtype=torch.float32)
    q, scales, zeros = dandan_ops.quantize_int4(weight, group_size=64)
    packed = dandan_ops.pack_int4(q)
    assert packed.shape == (3, 9)
    assert torch.equal(dandan_ops.unpack_int4(packed, original_k=65), q)
    restored = dandan_ops.dequantize_int4(packed, scales, zeros, 65, 64)
    assert restored.shape == weight.shape
    assert torch.max((restored - weight).abs()) <= scales.max() + 1e-5


def test_quantized_linear_cpu():
    weight = torch.randn(7, 65, dtype=torch.float32)
    bias = torch.randn(7, dtype=torch.float32)
    layer = dandan_ops.QuantizedLinear.from_float(weight, bias, group_size=64)
    x = torch.randn(4, 65, dtype=torch.float16)
    actual = layer(x)
    reference = x.float() @ dandan_ops.dequantize_int4(
        layer.packed_weight, layer.scales, layer.zeros, 65, 64
    ).half().float().t()
    reference = reference.half() + bias.half()
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    functional = dandan_ops.quantized_linear(
        x, layer.packed_weight, layer.scales, layer.zeros,
        in_features=65, group_size=64, bias=bias,
    )
    torch.testing.assert_close(functional, actual)
