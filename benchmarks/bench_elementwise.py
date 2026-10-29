"""Public small-op latency against unfused PyTorch, not engine speedup.

Report CUDA Graph GPU latency separately from amortized eager wall time.
Inputs repeat (warm cache); output allocation and layout conversion are
included. No model, scheduler, transfers, graph capture or compilation is timed.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
import dandan_ops as ops
if __package__:
    from .bench_hgemm import measure
else:
    from bench_hgemm import measure


def eager_measure(paths, repeats, batch):
    samples = {name: [] for name in paths}
    rng = random.Random(17)
    for _ in range(repeats):
        order = list(paths)
        rng.shuffle(order)
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(batch):
                paths[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / batch)
    return {name: {"median_us": statistics.median(values), "samples_us": values}
            for name, values in samples.items()}


@torch.inference_mode()
def run(args):
    torch.manual_seed(42)
    rows = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for shape in args.shapes:
            m, d = map(int, shape.lower().split("x"))
            if min(m, d) <= 0:
                raise ValueError("benchmark dimensions must be positive")
            x = torch.randn(m, d, device="cuda", dtype=dtype)
            residual = torch.randn_like(x)
            weight = torch.randn(d, device="cuda", dtype=dtype)
            for name, inputs in (("rms_norm", (x, weight)),
                                 ("residual_rms_norm", (x, residual, weight)),
                                 ("silu_and_mul", (x, residual))):
                fn = getattr(ops, name)
                paths = {"cuda": lambda: fn(*inputs, backend="cuda"),
                         "torch": lambda: fn(*inputs, backend="torch")}
                actual, expected = paths["cuda"](), paths["torch"]()
                actual = actual if isinstance(actual, tuple) else (actual,)
                expected = expected if isinstance(expected, tuple) else (expected,)
                tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
                errors = []
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
                    errors.append({"max_abs": (a.float()-b.float()).abs().max().item(),
                                   "relative_l2": ((a.float()-b.float()).norm() /
                                                   b.float().norm().clamp_min(1e-12)).item()})
                graph = measure(paths, args.warmup, args.repeats, args.batch)
                eager = eager_measure(paths, args.repeats, args.batch)
                row = {"op": name, "shape": [m, d], "dtype": dtype_name,
                       "execution_info": ops.elementwise_execution_info(name, *inputs, backend="cuda"),
                       "errors": errors, "graph_gpu": graph, "eager_amortized_wall": eager}
                rows.append(row)
                print(f"{name} {dtype_name} {shape}: graph cuda={graph['cuda']['median_us']:.2f}, "
                      f"torch={graph['torch']['median_us']:.2f} us", flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = ["dandan_ops/elementwise.py", "dandan_ops/csrc/elementwise.cu",
               "benchmarks/bench_elementwise.py"]
    report = {"gpu": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_capability(),
              "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
              "scope": __doc__, "config": vars(args),
              "source_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
              "results": rows}
    destination = Path(args.json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--shapes", nargs="+", default=["1x1024", "8x1024", "32x4096"])
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--json", default="build/benchmarks/bench_elementwise.json")
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.batch) < 1:
        parser.error("timing counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
