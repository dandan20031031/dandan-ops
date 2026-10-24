import pytest
import torch
import dandan_ops as ops
from dandan_ops._extension import load_extension


@pytest.fixture(autouse=True)
def cuda_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    load_extension(required=True)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_reduce = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(47)
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_reduce


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (8, 37, 65), (16, 63, 64),
    (17, 65, 129), (32, 128, 256), (33, 67, 128), (128, 64, 64),
    (129, 127, 257), (256, 128, 512)])
def test_dense_and_quantized_tails(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    expected = (a.float() @ w.float().t()).half()
    actual = ops.gemm(a, w, weight_transposed=True, backend="cuda")
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    # Standard GEMM layout and noncontiguous input both work via API copies.
    torch.testing.assert_close(ops.gemm(a, w.t(), backend="cuda"), actual)
    layer = ops.QuantizedLinear.from_float(w, backend="tensorcore")
    # Different oracle: tile weights are rounded to FP16 before MMA.
    wh = ops.dequantize_int4(layer.packed_weight, layer.scales, layer.zeros, k).half()
    expected = (a.float() @ wh.float().t()).half()
    torch.testing.assert_close(layer(a), expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("m,n,k", [(128, 64, 64), (256, 128, 512), (512, 192, 1024)])
def test_original_tma(m, n, k):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 TMA backend")
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    expected = (a.float() @ w.float().t()).half()
    torch.testing.assert_close(ops.gemm(a, w, weight_transposed=True, backend="tma"),
                               expected, atol=2e-3, rtol=2e-3)


def test_nondefault_stream_and_alignment():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.randn(1 + 17 * 65, device="cuda", dtype=torch.float16)[1:].view(17, 65)
        w = torch.randn(1 + 67 * 65, device="cuda", dtype=torch.float16)[1:].view(67, 65)
        dense = ops.gemm(a, w, weight_transposed=True, backend="cuda")
        layer = ops.QuantizedLinear.from_float(w, backend="tensorcore")
        quant = layer(a)
        wh = ops.dequantize_int4(layer.packed_weight, layer.scales, layer.zeros, 65).half()
        dense_ref = (a.float() @ w.float().t()).half()
        quant_ref = (a.float() @ wh.float().t()).half()
    stream.synchronize()
    torch.testing.assert_close(dense, dense_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(quant, quant_ref, atol=2e-3, rtol=2e-3)


def test_routes_and_no_full_weight_dequant(monkeypatch):
    import importlib
    mod = importlib.import_module("dandan_ops.quantized_linear")
    layer = ops.QuantizedLinear.from_float(torch.ones(67, 129, device="cuda", dtype=torch.float16))
    def forbidden(*args, **kwargs):
        pytest.fail("supported CUDA forward must not materialize full FP16/FP32 weight")
    monkeypatch.setattr(mod, "dequantize_int4", forbidden)
    for m, name in [(1, "w4a16_gemv"), (8, "w4a16_tensorcore")]:
        from unittest.mock import Mock
        original = getattr(torch.ops.dandan_ops, name)
        selected = Mock(wraps=original)
        with monkeypatch.context() as patch:
            patch.setattr(torch.ops.dandan_ops, name, selected)
            actual = layer(torch.ones(m, 129, device="cuda", dtype=torch.float16))
            selected.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 129))


def test_native_validation_and_empty_batch():
    layer = ops.QuantizedLinear.from_float(torch.randn(5, 65, device="cuda", dtype=torch.float16))
    a = torch.randn(8, 65, device="cuda", dtype=torch.float16)
    p, s, z = layer.packed_weight, layer.scales, layer.zeros
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.dandan_ops.w4a16_tensorcore(a.t(), p, s, z, 64)
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        torch.ops.dandan_ops.w4a16_gemv(a[:1], p.cpu(), s, z, 64)
    with pytest.raises(RuntimeError, match="metadata"):
        torch.ops.dandan_ops.w4a16_tensorcore(a, p, s[:, :1].contiguous(), z, 64)
    assert layer(a[:0]).shape == (0, 5)
    with pytest.raises(RuntimeError, match="M%128"):
        ops.gemm(a, torch.randn(64, 65, device="cuda", dtype=torch.float16),
                 weight_transposed=True, backend="tma")


def test_cpu_and_autograd_fallback():
    # CPU is tested outside this fixture by the existing API suite as well.
    a = torch.randn(2, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(67, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    y = ops.gemm(a, w, weight_transposed=True)
    y.sum().backward()
    assert a.grad is not None and w.grad is not None


def test_auto_tma_alignment_and_fallback(monkeypatch):
    from unittest.mock import Mock
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 auto TMA")
    a = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    w = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    selected = Mock(wraps=torch.ops.dandan_ops.hgemm_tma)
    monkeypatch.setattr(torch.ops.dandan_ops, "hgemm_tma", selected)
    actual = ops.gemm(a, w, weight_transposed=True)
    selected.assert_called_once()
    torch.testing.assert_close(actual, (a.float() @ w.float().t()).half(), atol=2e-3, rtol=2e-3)
    selected.reset_mock()
    unaligned = torch.randn(a.numel() + 1, device="cuda", dtype=torch.float16)[1:].view_as(a)
    actual = ops.gemm(unaligned, w, weight_transposed=True)
    selected.assert_not_called()
    torch.testing.assert_close(actual, (unaligned.float() @ w.float().t()).half(), atol=2e-3, rtol=2e-3)
