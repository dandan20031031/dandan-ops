"""Numerical contracts, routing, storage boundaries and CUDA Graph safety."""
import warnings

import pytest
import torch
import torch.nn.functional as F

import dandan_ops as ops
from dandan_ops._extension import load_extension


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and (not torch.cuda.is_available() or not load_extension()):
        pytest.skip("built CUDA extension required")
    return request.param


@pytest.fixture
def gpu():
    if not torch.cuda.is_available() or not load_extension():
        pytest.skip("built CUDA extension required")
    return "cuda"


def rms_oracle(x, w, eps=1e-5):
    # Double precision oracle, with the public low-precision rounding point.
    work = x.double()
    return (work * (work.square().mean(-1, keepdim=True) + eps).rsqrt()).to(x.dtype) * w


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(8,), (3, 257), (2, 3, 1024), (0, 8)])
def test_rms_dtype_shape_and_fp32_accumulation(device, dtype, shape):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype) * 300
    w = torch.randn(shape[-1], device=device, dtype=dtype)
    actual = ops.rms_norm(x, w)
    tol = 2e-2 if dtype == torch.bfloat16 else (2e-3 if dtype == torch.float16 else 2e-6)
    torch.testing.assert_close(actual, rms_oracle(x, w), atol=tol, rtol=tol)
    assert actual.shape == x.shape and actual.dtype == x.dtype


def test_rms_half_overflow_regression(device):
    x = torch.full((1, 8), 300., dtype=torch.float16, device=device)
    w = torch.ones(8, dtype=x.dtype, device=device)
    torch.testing.assert_close(ops.rms_norm(x, w), torch.ones_like(x), atol=0, rtol=0)


@pytest.mark.parametrize("offset", [1000., 10000.])
def test_layernorm_cancellation_regression(device, offset):
    x = torch.tensor([[-.3, -.2, -.1, 0., 0., .1, .2, .3]], device=device) + offset
    w = torch.ones(8, device=device)
    b = torch.zeros_like(w)
    expected = F.layer_norm(x.double(), (8,), w.double(), b.double(), 1e-5).float()
    if device == "cpu":
        # CPU is delegated to PyTorch; its FP32 implementation has its own
        # rounding error for large offsets. The CUDA kernel uses the FP64 oracle.
        expected = F.layer_norm(x, (8,), w, b, 1e-5)
    torch.testing.assert_close(ops.layer_norm(x, w, b), expected, atol=2e-5, rtol=2e-5)


def case(name, device, requires_grad=False):
    lowp = name in ("residual_rms_norm", "silu_and_mul")
    dtype = torch.float16 if lowp else torch.float32
    x = torch.randn(3, 8, device=device, dtype=dtype, requires_grad=requires_grad)
    w = torch.randn(8, device=device, dtype=dtype, requires_grad=requires_grad)
    b = torch.randn(8, device=device, dtype=dtype, requires_grad=requires_grad)
    r = torch.randn_like(x, requires_grad=requires_grad)
    return {"softmax": (x,), "rms_norm": (x, w), "layer_norm": (x, w, b),
            "batch_norm_infer": (x, w, b), "residual_rms_norm": (x, r, w),
            "silu_and_mul": (x, r)}[name]


NAMES = ["softmax", "rms_norm", "layer_norm", "batch_norm_infer", "residual_rms_norm", "silu_and_mul"]


def outputs(value):
    return value if isinstance(value, (tuple, list)) else (value,)


@pytest.mark.parametrize("name", NAMES)
def test_autograd_falls_back_and_preserves_all_input_gradients(device, name):
    args = case(name, device, requires_grad=True)
    reference_args = [x.detach().clone().requires_grad_() for x in args]
    fn = getattr(ops, name)
    with warnings.catch_warnings(record=True) as caught:
        y = outputs(fn(*args))
        ref = outputs(fn(*reference_args, backend="torch"))
        multipliers = [torch.randn_like(t) for t in y]
        sum((t * m).sum() for t, m in zip(y, multipliers)).backward()
        sum((t * m).sum() for t, m in zip(ref, multipliers)).backward()
    assert not any("autograd kernel was not registered" in str(w.message) for w in caught)
    for a, b in zip(args, reference_args):
        assert a.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    assert ops.elementwise_execution_info(name, *args)["implementation"] == "torch"
    with pytest.raises(ValueError):
        fn(*args, backend="cuda")


@pytest.mark.parametrize("name", NAMES)
def test_explicit_backend_contract_and_empty_batch(device, name):
    args = case(name, device)
    args = tuple(t[:0] if t.ndim == 2 else t for t in args)
    y = outputs(getattr(ops, name)(*args))
    assert all(t.shape == (0, 8) for t in y)
    assert ops.elementwise_execution_info(name, *args, backend="torch")["implementation"] == "torch"
    with pytest.raises(ValueError, match="backend"):
        getattr(ops, name)(*args, backend="invalid")
    if device == "cpu":
        with pytest.raises(ValueError, match="CUDA"):
            getattr(ops, name)(*args, backend="cuda")


@pytest.mark.parametrize("eps", [-1., float("nan"), float("inf"), 1e100])
def test_invalid_eps(device, eps):
    x, w = case("rms_norm", device)
    with pytest.raises(ValueError, match="eps"):
        ops.rms_norm(x, w, eps)
    with pytest.raises(ValueError, match="eps"):
        ops.layer_norm(x, w, w, eps)


def test_invalid_shapes_and_scalar_softmax(device):
    x = torch.ones(2, 8, device=device)
    with pytest.raises(ValueError, match="parameters"):
        ops.rms_norm(x, torch.ones(7, device=device))
    with pytest.raises(ValueError, match="last dimension"):
        ops.rms_norm(x[:, :0], torch.ones(0, device=device))
    with pytest.raises(ValueError, match="shape and dtype"):
        ops.silu_and_mul(x, x.half())
    with pytest.raises(TypeError, match="floating"):
        ops.softmax(x.int())
    with pytest.raises(IndexError):
        ops.softmax(x, dim=2)
    scalar = torch.tensor(2., device=device)
    torch.testing.assert_close(ops.softmax(scalar), torch.ones_like(scalar))
    torch.testing.assert_close(ops.softmax(x, dim=0), F.softmax(x, dim=0))
    torch.testing.assert_close(ops.softmax(x[:, :0]), x[:, :0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (3, 257), (2, 3, 1024)])
def test_fused_rounding_and_no_mutation(gpu, dtype, shape):
    torch.manual_seed(123)
    x = torch.randn(shape, device=gpu, dtype=dtype)
    r = torch.randn_like(x)
    w = torch.randn(shape[-1], device=gpu, dtype=dtype)
    saved = [t.clone() for t in (x, r, w)]
    y, summed = ops.residual_rms_norm(x, r, w, backend="cuda")
    torch.testing.assert_close(summed, x + r, atol=0, rtol=0)
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(y, rms_oracle(x + r, w), atol=tol, rtol=tol)
    torch.testing.assert_close(ops.silu_and_mul(x, r, backend="cuda"), F.silu(x) * r,
                               atol=tol, rtol=tol)
    for actual, expected in zip((x, r, w), saved):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert y.data_ptr() != x.data_ptr() and summed.data_ptr() != r.data_ptr()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("layout", ["offset", "strided"])
def test_layout_graph_and_nondefault_stream(gpu, name, layout):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        args = case(name, gpu)
        def changed(t):
            if layout == "offset":
                backing = torch.empty(t.numel() + 1, device=gpu, dtype=t.dtype)
                view = backing[1:].view(t.shape)
            else:
                view = torch.empty((*t.shape, 2), device=gpu, dtype=t.dtype)[..., 0]
            view.copy_(t)
            return view
        args = tuple(changed(t) for t in args)
        info = ops.elementwise_execution_info(name, *args, backend="cuda")
        if layout == "strided" or name == "rms_norm":
            assert info["input_copies"] == list(range(len(args)))
        else:
            assert info["input_copies"] == []
        fn = getattr(ops, name)
        expected = outputs(fn(*args, backend="torch"))
        for _ in range(3):
            fn(*args, backend="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = outputs(fn(*args, backend="cuda"))
        graph.replay()
        first = [t.clone() for t in actual]
        args[0].add_(0.1)
        second_expected = outputs(fn(*args, backend="torch"))
        graph.replay()
        second = [t.clone() for t in actual]
    stream.synchronize()
    for result, reference in zip(first + second, list(expected) + list(second_expected)):
        torch.testing.assert_close(result, reference, atol=2e-3, rtol=2e-3)


def test_native_guards_and_unavailable_extension(gpu, monkeypatch):
    import dandan_ops.elementwise as implementation
    x, w = case("rms_norm", gpu)
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.dandan_ops.rms_norm(x[:, ::2], w[:4], 1e-5)
    backing = torch.ones(17, device=gpu)
    with pytest.raises(RuntimeError, match="alignment"):
        torch.ops.dandan_ops.rms_norm(backing[1:].view(2, 8), w, 1e-5)
    with pytest.raises(RuntimeError, match="inference-only"):
        torch.ops.dandan_ops.softmax(x.requires_grad_())
    x = x.detach()
    with pytest.raises(ValueError, match="device"):
        ops.rms_norm(x, w.cpu())
    with pytest.raises(RuntimeError, match="device"):
        torch.ops.dandan_ops.rms_norm(x, w.cpu(), 1e-5)
    monkeypatch.setattr(implementation, "load_extension", lambda required=False: False)
    report = ops.elementwise_execution_info("rms_norm", x, w)
    assert report["implementation"] == "torch" and report["reason"] == "CUDA extension unavailable"
    torch.testing.assert_close(ops.rms_norm(x, w), rms_oracle(x, w))


def test_softmax_nonfinite_and_tail_columns(gpu):
    x = torch.randn(3, 259, device=gpu)
    x[0, 0] = float("nan")
    x[1].fill_(-float("inf"))
    x[2, 0] = float("inf")
    torch.testing.assert_close(ops.softmax(x, backend="cuda"), F.softmax(x, -1), equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lowp_rms_graph(gpu, dtype):
    x = torch.randn(3, 257, dtype=dtype, device=gpu)
    w = torch.randn(257, dtype=dtype, device=gpu)
    for _ in range(3):
        ops.rms_norm(x, w, backend="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ops.rms_norm(x, w, backend="cuda")
    graph.replay()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(output, rms_oracle(x, w), atol=tolerance, rtol=tolerance)


def test_execution_info_copies_for_gemm_and_attention(gpu):
    a = torch.randn(8, 16, device=gpu, dtype=torch.float16)
    b = torch.randn(16, 24, device=gpu, dtype=torch.float16)
    assert ops.gemm_execution_info(a, b, backend="cuda")["input_copies"] == [1]
    q = torch.randn(1, 2, 8, 128, device=gpu, dtype=torch.float16)
    report = ops.flash_attention_execution_info(q[:, :, ::2], q, q, backend="cuda")
    assert report["input_copies"] == [0] and report["implementation"] == "cuda"
    assert ops.flash_attention_execution_info(q, q, q, backend="torch")["input_copies"] == []
