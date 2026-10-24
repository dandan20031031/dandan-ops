import pytest
import torch
import torch.nn.functional as F
import dandan_ops

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(q, k, v, causal, start, scale=None):
    mask = None
    if causal:
        mask = (torch.arange(k.shape[2], device=q.device)[None, :]
                <= torch.arange(q.shape[2], device=q.device)[:, None] + start)
    # FP32 math with an explicit mask is independent of the custom fragment
    # mapping, softmax conversion to FP16, and SDPA's rectangular-causal rule.
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=mask, scale=scale, enable_gqa=True).to(q.dtype)


@cuda_only
@pytest.mark.parametrize("B,Hq,Hkv,Q,K,start", [
    (1, 1, 1, 1, 1, 0),
    (1, 16, 8, 1, 257, 256),
    (2, 4, 2, 1, 1024, 1023),
    (1, 4, 1, 1, 4097, 4096),
    (1, 4, 2, 1, 257, 0),
    (1, 4, 2, 1, 257, 13),
    (1, 4, 4, 5, 17, 0),
    (1, 16, 8, 5, 17, 12),
    (1, 16, 8, 17, 17, 0),
    (2, 4, 2, 33, 65, 0),
    (1, 4, 2, 64, 65, 1),
    (1, 4, 2, 65, 257, 192),
    (1, 4, 2, 127, 257, 0),
    (1, 16, 8, 128, 128, 0),
    (1, 4, 2, 128, 1024, 896),
    (1, 4, 1, 129, 257, 128),
])
@pytest.mark.parametrize("causal", [True, False])
@torch.inference_mode()
def test_flash_inference_matches_sdpa(B, Hq, Hkv, Q, K, start, causal):
    torch.manual_seed(B + Hq + Hkv + Q + K + start)
    q = torch.randn(B, Hq, Q, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, K, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    actual = dandan_ops.flash_attention(q, k, v, causal=causal, backend="cuda", query_start=start)
    expected = reference(q, k, v, causal, start)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@cuda_only
@pytest.mark.parametrize("Q,K", [(1, 257), (17, 257)])
@pytest.mark.parametrize("scale", [0.0, 0.17, -0.1])
@torch.inference_mode()
def test_custom_scale(Q, K, scale):
    q = torch.randn(1, 4, Q, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, K, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    actual = dandan_ops.flash_attention(q, k, v, causal=True, scale=scale,
                                       query_start=K-Q, backend="cuda")
    torch.testing.assert_close(actual, reference(q, k, v, True, K-Q, scale), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("backend", ["auto", "torch"])
@pytest.mark.parametrize("scale", [None, 0.25])
def test_cpu_gqa_suffix_fallback(backend, scale):
    q = torch.randn(1, 4, 3, 16)
    k = torch.randn(1, 2, 9, 16)
    v = torch.randn_like(k)
    actual = dandan_ops.flash_attention(q, k, v, causal=True, query_start=6,
                                       scale=scale, backend=backend)
    torch.testing.assert_close(actual, reference(q, k, v, True, 6, scale))


@pytest.mark.parametrize("start", [-1, 5, 0.5, True])
def test_invalid_start(start):
    q = torch.randn(1, 2, 3, 16)
    with pytest.raises(ValueError, match="query_start"):
        dandan_ops.flash_attention(q, q, q, query_start=start)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 1e100])
def test_invalid_scale(scale):
    q = torch.randn(1, 2, 3, 16)
    with pytest.raises(ValueError, match="scale"):
        dandan_ops.flash_attention(q, q, q, scale=scale)


def test_flash_api_rejects_invalid_shape_and_backend():
    q = torch.randn(1, 2, 4, 64)
    with pytest.raises(ValueError, match="backend"):
        dandan_ops.flash_attention(q, q, q, backend="bad")
    with pytest.raises(ValueError, match="non-empty"):
        dandan_ops.flash_attention(q[:, :0], q[:, :0], q[:, :0])
    with pytest.raises(ValueError, match="dtype and device"):
        dandan_ops.flash_attention(q, q.half(), q.half())
    with pytest.raises(ValueError, match="GQA"):
        dandan_ops.flash_attention(q, q.repeat(1, 2, 1, 1), q.repeat(1, 2, 1, 1))
    with pytest.raises(ValueError, match="FP16"):
        dandan_ops.flash_attention(q, q, q, backend="cuda")


@cuda_only
@pytest.mark.parametrize("layout", ["strided", "unaligned"])
@torch.inference_mode()
def test_layouts(layout):
    def make(heads, length):
        shape = (1, heads, length, 128)
        if layout == "strided":
            return torch.randn(1, heads, 128, length, device="cuda", dtype=torch.float16).transpose(-1, -2)
        return torch.randn(heads*length*128+1, device="cuda", dtype=torch.float16)[1:].view(shape)
    q, k, v = make(4, 17), make(2, 65), make(2, 65)
    actual = dandan_ops.flash_attention(q, k, v, causal=True, query_start=48, backend="cuda")
    torch.testing.assert_close(actual, reference(q, k, v, True, 48), atol=2e-3, rtol=2e-3)


@cuda_only
@pytest.mark.parametrize("Q", [1, 17])
@torch.inference_mode()
def test_graph_and_nondefault_stream(Q):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q = torch.randn(1, 4, Q, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        fn = lambda: dandan_ops.flash_attention(q, k, v, causal=True, query_start=257-Q, backend="cuda")
        for _ in range(3): fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fn()
    q.add_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference(q, k, v, True, 257-Q), atol=2e-3, rtol=2e-3)


@cuda_only
def test_cuda_grad_and_dtype_routing():
    q = torch.randn(1, 4, 3, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 9, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    with pytest.raises(ValueError, match="forward inference"):
        dandan_ops.flash_attention(q, k, v, backend="cuda")
    actual = dandan_ops.flash_attention(q, k, v, causal=True, query_start=6)
    actual.float().square().mean().backward()
    for x in (q, k, v): assert x.grad is not None and torch.isfinite(x.grad).all()
    with torch.inference_mode():
        for backend in ("torch", "auto"):
            actual = dandan_ops.flash_attention(q.float(), k.float(), v.float(),
                                               causal=True, query_start=6, backend=backend)
            torch.testing.assert_close(actual, reference(q.float(), k.float(), v.float(), True, 6))
        with pytest.raises(ValueError, match="FP16"):
            dandan_ops.flash_attention(q.float(), k.float(), v.float(), backend="cuda")


@cuda_only
@pytest.mark.parametrize("Q", [1, 17])
@torch.inference_mode()
def test_causal_hides_future_and_large_logits(Q):
    q = torch.full((1, 4, Q, 128), 32.0, device="cuda", dtype=torch.float16)
    k = torch.full((1, 2, 257, 128), 32.0, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    actual = dandan_ops.flash_attention(q, k, v, causal=True, backend="cuda")
    expected = reference(q, k, v, True, 0)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    k[:, :, Q:] = -64
    v[:, :, Q:] = 60000
    again = dandan_ops.flash_attention(q, k, v, causal=True, backend="cuda")
    torch.testing.assert_close(again, actual, atol=0, rtol=0)
