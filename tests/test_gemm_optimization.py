"""Numerics, tails, stream/graph lifetime and dispatch safety for new routes."""
import pytest
import torch
import dandan_ops as ops

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
@pytest.mark.parametrize("m,n,k", [(0,5,65),(1,1,1),(2,5,7),(3,7,9),(4,31,63),
    (5,33,65),(7,5,129),(8,37,257),(1,2048,1024),(2,1024,3072),(4,3072,1024),(8,1024,1024)])
@torch.inference_mode()
def test_small_w4_tails(m,n,k):
    torch.manual_seed(m+n+k)
    layer = ops.QuantizedLinear.from_float(torch.randn(n,k,device="cuda",dtype=torch.float16), backend="small")
    x = torch.randn(m,k,device="cuda",dtype=torch.float16)
    expected = (x.float() @ layer.dequantize().half().float().t()).half()
    torch.testing.assert_close(layer(x), expected, atol=2e-3, rtol=2e-3)


@cuda_only
@pytest.mark.parametrize("n,k", [(5,1),(7,9),(37,65),(31,257),(2048,1024),(1024,3072),(6144,1024)])
@torch.inference_mode()
def test_small_single_row_preserves_original_gemv_tree(n,k):
    torch.manual_seed(n+k)
    layer = ops.QuantizedLinear.from_float(torch.randn(n,k,device="cuda",dtype=torch.float16))
    x = torch.randn(1,k,device="cuda",dtype=torch.float16)
    layer.backend = "gemv"
    expected = layer(x)
    layer.backend = "small"
    torch.testing.assert_close(layer(x),expected,atol=0,rtol=0)


@cuda_only
@torch.inference_mode()
def test_small_w4_nondefault_stream_graph_unaligned():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer = ops.QuantizedLinear.from_float(torch.randn(37,256,device="cuda",dtype=torch.float16), backend="small")
        # Contiguous but misaligned takes the scalar masked load path.
        x = torch.randn(8*256+1,device="cuda",dtype=torch.float16)[1:].view(8,256)
        for _ in range(3): layer(x)
        expected = (x.float() @ layer.dequantize().half().float().t()).half()
        actual = layer(x)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = layer(x)
    x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    with pytest.raises(ValueError, match="M<=8"):
        layer(torch.ones(9,256,device="cuda",dtype=torch.float16))
    with pytest.raises(RuntimeError, match="M<=8"):
        torch.ops.dandan_ops.w4a16_small(torch.ones(9,256,device="cuda",dtype=torch.float16),
            layer.packed_weight,layer.scales,layer.zeros,64)


@cuda_only
@torch.inference_mode()
def test_tma_fp16_graph_nondefault_stream():
    if torch.cuda.get_device_capability() != (12,0): pytest.skip("SM120")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        a = torch.randn(128,1024,device="cuda",dtype=torch.float16)
        w = torch.randn(2048,1024,device="cuda",dtype=torch.float16)
        for _ in range(3): ops.gemm(a,w,weight_transposed=True,backend="tma")
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = ops.gemm(a,w,weight_transposed=True,backend="tma")
    a.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert output.dtype == torch.float16
    torch.testing.assert_close(output,(a.float() @ w.float().t()).half(),atol=2e-3,rtol=2e-3)


def test_profile_mismatched_environment_shape_and_invalid_route(monkeypatch):
    import dandan_ops._dispatch as dispatch
    env = {"gpu":"test", "sm":[12,0], "torch":"2.9.1+cu128", "torch_cuda":"12.8"}
    monkeypatch.setattr(dispatch, "_profile", {"dispatch_revision":1, "environment":env,
        "routes":{"dense":{"1x2048x1024":"torch", "2x2048x1024":"invalid"}}})
    monkeypatch.setattr(dispatch, "_device_signature", lambda index: env)
    device = torch.device("cuda:0")
    assert dispatch.measured_route("dense",1,2048,1024,device) == "torch"
    assert dispatch.measured_route("dense",2,2048,1024,device) is None
    assert dispatch.measured_route("dense",3,2048,1024,device) is None
    monkeypatch.setattr(dispatch, "_device_signature", lambda index: {**env, "torch_cuda":"13.0"})
    assert dispatch.measured_route("dense",1,2048,1024,device) is None


@cuda_only
@torch.inference_mode()
def test_measured_routes_no_full_dequant_and_explicit_overrides(monkeypatch):
    import importlib
    quant = importlib.import_module("dandan_ops.quantized_linear")
    import dandan_ops._dispatch as dispatch
    env = dispatch._device_signature(torch.cuda.current_device())
    monkeypatch.setattr(dispatch, "_profile", {"dispatch_revision":1,"environment":env,
        "routes":{"dense":{"2x37x65":"torch"},"w4a16":{"2x37x65":"w4a16_small"}}})
    w = torch.randn(37,65,device="cuda",dtype=torch.float16)
    x = torch.randn(2,65,device="cuda",dtype=torch.float16)
    layer = ops.QuantizedLinear.from_float(w)
    expected = (x.float() @ layer.dequantize().half().float().t()).half()
    def forbidden(*a,**kw): pytest.fail("unexpected full-weight dequantization or custom dense GEMM")
    monkeypatch.setattr(quant, "dequantize_int4", forbidden)
    assert ops.gemm_execution_info(x,w,weight_transposed=True)["implementation"] == "torch"
    assert ops.gemm_execution_info(x,w,weight_transposed=True,backend="cuda")["implementation"] == "cuda"
    assert layer.execution_info(x)["implementation"] == "w4a16_small"
    torch.testing.assert_close(layer(x), expected, atol=2e-3,rtol=2e-3)
    monkeypatch.setattr(torch.ops.dandan_ops, "hgemm", forbidden)
    torch.testing.assert_close(ops.gemm(x,w,weight_transposed=True), x @ w.t())
