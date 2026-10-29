"""Install a wheel in an isolated temporary target and validate GPU inference.

Usage: python scripts/verify_gpu_wheel.py build/wheels/<wheel>.whl
Uses the invoking interpreter's PyTorch; never installs engine dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


SMOKE = r'''
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import torch
import dandan_ops as ops
from dandan_ops._extension import load_extension
assert Path(ops.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert "mini_engine" not in sys.modules
assert torch.cuda.is_available(), "GPU release check requires CUDA"
load_extension(required=True)
assert torch.ops.dandan_ops.w4a16_contract_version() == 2
assert torch.ops.dandan_ops.elementwise_contract_version() == 1
assert torch.ops.dandan_ops.attention_contract_version() == 1
with torch.inference_mode():
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.full((2, 1024), 300., device="cuda", dtype=dtype)
        w = torch.ones(1024, device="cuda", dtype=dtype)
        r = torch.ones_like(x)
        y, summed = ops.residual_rms_norm(x, r, w, backend="cuda")
        torch.testing.assert_close(y, torch.ones_like(x), atol=0, rtol=0)
        torch.testing.assert_close(summed, x + r, atol=0, rtol=0)
        torch.testing.assert_close(ops.rms_norm(x, w, backend="cuda"), torch.ones_like(x))
        torch.testing.assert_close(ops.silu_and_mul(x, r, backend="cuda"),
                                   torch.nn.functional.silu(x) * r)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_y, graph_sum = ops.residual_rms_norm(x, r, w, backend="cuda")
        graph.replay()
        torch.testing.assert_close(graph_y, y)
        torch.testing.assert_close(graph_sum, summed)
    x = torch.tensor([[999.7, 999.8, 999.9, 1000., 1000., 1000.1, 1000.2, 1000.3]], device="cuda")
    w = torch.ones(8, device="cuda"); b = torch.zeros_like(w)
    expected = torch.nn.functional.layer_norm(x.double(), (8,), w.double(), b.double(), 1e-5).float()
    torch.testing.assert_close(ops.layer_norm(x,w,b,backend="cuda"), expected, atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(ops.softmax(x,backend="cuda"), torch.softmax(x,-1))
    torch.testing.assert_close(ops.batch_norm_infer(x,w,b,backend="cuda"), x)
    a = torch.randn(2,64,device="cuda",dtype=torch.float16)
    weight = torch.randn(32,64,device="cuda",dtype=torch.float16)
    torch.testing.assert_close(ops.gemm(a,weight,weight_transposed=True,backend="cuda"),
                               a @ weight.t(),atol=2e-3,rtol=2e-3)
    layer = ops.QuantizedLinear.from_float(weight, backend="cuda")
    torch.testing.assert_close(layer(a), ops.quantized_linear(a,layer.packed_weight,layer.scales,
        layer.zeros,in_features=64,backend="reference"),atol=2e-3,rtol=2e-3)
    q = torch.randn(1,2,8,128,device="cuda",dtype=torch.float16)
    torch.testing.assert_close(ops.flash_attention(q,q,q,backend="cuda"),
                               ops.flash_attention(q,q,q,backend="torch"),atol=2e-3,rtol=2e-3)
    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(1,4,17,256,device="cuda",dtype=dtype)
        k = torch.randn(1,2,33,256,device="cuda",dtype=dtype); v = torch.randn_like(k)
        tol = 1.6e-2 if dtype == torch.bfloat16 else 2e-3
        kwargs = dict(causal=True,query_start=16,window_size=7,softcap=4.)
        torch.testing.assert_close(ops.flash_attention(q,k,v,backend="cuda",**kwargs),
                                   ops.flash_attention(q,k,v,backend="torch",**kwargs),atol=tol,rtol=tol)
        plan = ops.prepare_varlen_attention([17],[33],[16],device="cuda")
        qp=q[0].transpose(0,1).contiguous();kp=k[0].transpose(0,1).contiguous();vp=v[0].transpose(0,1).contiguous()
        torch.testing.assert_close(ops.flash_attention_varlen(qp,kp,vp,plan,backend="cuda"),
                                   ops.flash_attention_varlen(qp,kp,vp,plan,backend="torch"),atol=tol,rtol=tol)
        cache=torch.randn(3,16,2,256,device="cuda",dtype=dtype)
        plan=ops.prepare_paged_attention([[2,0]],[17],num_blocks=3,block_size=16,device="cuda")
        query=q[:,:,0].contiguous()
        torch.testing.assert_close(ops.paged_attention(query,cache,cache,plan,backend="cuda"),
                                   ops.paged_attention(query,cache,cache,plan,backend="torch"),atol=tol,rtol=tol)
x = torch.randn(2,8,device="cuda",requires_grad=True)
ops.softmax(x).square().sum().backward()
assert x.grad is not None and x.grad.isfinite().all()
torch.cuda.synchronize()
print(json.dumps({"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),
    "torch_cuda":torch.version.cuda,"elementwise_contract":1,"w4a16_contract":2,"attention_contract":1,
    "checks":["isolated wheel import","FP16/BF16 fused ops","RMS overflow","LayerNorm cancellation",
              "CUDA Graph","Softmax autograd fallback","GEMM","W4A16","Attention FP16/BF16 D256",
              "packed prefill","direct paged decode"]}))
'''


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = args.wheel.resolve(strict=True)
    sources = ["dandan_ops/elementwise.py", "dandan_ops/functional.py", "dandan_ops/attention.py",
               "dandan_ops/_extension.py", "dandan_ops/__init__.py"]
    with zipfile.ZipFile(wheel) as archive:
        for name in sources:
            if archive.read(name) != (root/name).read_bytes():
                raise RuntimeError(f"stale wheel source: {name}")
    temporary = (root / "build/tmp").resolve()
    temporary.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="gpu-wheel-", dir=temporary)).resolve()
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                        "--target", str(target), str(wheel)], check=True)
        result = subprocess.run([sys.executable, "-I", "-c", SMOKE, str(target)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        report.update(wheel=wheel.name, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      matching_sources=sources)
        (root / "build/wheel_gpu_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if not target.is_relative_to(temporary) or target == temporary:
            raise RuntimeError("unsafe temporary cleanup path")
        shutil.rmtree(target)


if __name__ == "__main__":
    main()
