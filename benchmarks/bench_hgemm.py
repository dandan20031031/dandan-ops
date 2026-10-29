"""Compare V8/V10 reuse and fused W4A16 with CUDA Graph replay timing.

Prequantization is outside timing. All paths have their normal output
allocation captured. Timings are per-call graph replay GPU latency, not Python
API wall time or end-to-end serving latency. Inputs repeat (warm cache).
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import dandan_ops as ops
from dandan_ops._extension import load_extension


def measure(paths, warmup, repeats, batch):
    graphs = {}
    for name, fn in paths.items():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Capture GPU work only; Python dispatch and first allocations are outside.
        with torch.cuda.graph(graph):
            outputs = [fn() for _ in range(batch)]
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        graphs[name] = (graph, outputs)
    samples = {name: [] for name in paths}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rng = random.Random(12)
    for _ in range(repeats):
        order = list(paths)
        rng.shuffle(order)
        for name in order:
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / batch)
    return {name: {"median_us": statistics.median(values), "samples_us": values}
            for name, values in samples.items()}


@torch.inference_mode()
def run(m, n, k, args):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
    w = torch.randn(n, k, dtype=torch.float16, device="cuda")
    layer = ops.QuantizedLinear.from_float(w)
    packed, scales, zeros = layer.packed_weight, layer.scales, layer.zeros
    dense_oracle = (x.float() @ w.float().t()).half()
    restored = ops.dequantize_int4(packed, scales, zeros, k)
    mma_oracle = (x.float() @ restored.half().float().t()).half()
    standard_b = w.t().contiguous()
    paths = {
        "torch_fp16": lambda: x @ w.t(),
        "hgemm_v8_weight_NK": lambda: ops.gemm(x, w, weight_transposed=True, backend="cuda"),
        "hgemm_v8_input_KN_with_copy": lambda: ops.gemm(x, standard_b, backend="cuda"),
        "w4a16_auto": lambda: layer(x),
        "w4a16_old_naive": lambda: ops.quantized_linear(x, packed, scales, zeros,
                            in_features=k, backend="naive"),
    }
    if m % 128 == 0 and n % 64 == 0 and k % 64 == 0:
        paths["hgemm_v10_tma_weight_NK"] = lambda: ops.gemm(x, w, weight_transposed=True, backend="tma")
    errors = {}
    for name, fn in paths.items():
        actual = fn()
        if name.startswith("hgemm") or name == "torch_fp16":
            expected = dense_oracle
        else:
            expected = mma_oracle
        # Scale input norms so an absolute tolerance remains meaningful over long K.
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        errors[name] = {"max_abs": (actual.float() - expected.float()).abs().max().item(),
                       "relative_l2": ((actual.float() - expected.float()).norm()
                                        / expected.float().norm()).item()}
    timing = measure(paths, args.warmup, args.repeats, args.batch)
    for name, fn in paths.items():
        # Validate again after warmup and capture.
        assert torch.isfinite(fn()).all(), name
    old = timing["w4a16_old_naive"]["median_us"]
    new = timing["w4a16_auto"]["median_us"]
    result = {"shape_MNK": [m, n, k], "timing": timing, "errors": errors,
              "old_over_new": old / new,
              "fp16_weight_mib": w.numel() * 2 / 2**20,
              "int4_state_mib": sum(t.numel() * t.element_size()
                                      for t in (packed, scales, zeros)) / 2**20}
    print(f"{m}x{n}x{k}: " + ", ".join(f"{name}={v['median_us']:.2f}us" for name,v in timing.items()), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--shapes", nargs="+", default=["1x4096x4096", "8x4096x4096",
        "32x4096x4096", "128x4096x4096", "512x4096x4096"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--json", default="build/benchmarks/bench_hgemm.json")
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.batch) < 1:
        parser.error("warmup/repeats/batch must be positive")
    load_extension(required=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    result = {"gpu": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "timing_mode": "CUDA Graph replay GPU latency per call; warm cache",
              "config": vars(args), "results": []}
    for shape in args.shapes:
        m, n, k = map(int, shape.lower().split("x"))
        result["results"].append(run(m, n, k, args))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
