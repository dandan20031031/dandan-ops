"""Qwen3 projection GPU latency; synthetic inputs, CUDA Graph, warm cache.

Derives N/K from a local model config. Quantization and full-weight FP32
oracles are outside timing. FP16 and W4A16 have separate numeric oracles.
The quantized paths retain packed weights; no dequantized-weight cache.
"""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import dandan_ops as ops
from bench_hgemm import measure


def measure_cold(paths, warmup, repeats, batch):
    if batch != 1:
        raise ValueError("--cold requires --batch 1 (each call must start after eviction)")
    graphs, outputs = {}, {}
    for name, fn in paths.items():
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): outputs[name] = fn()
        graphs[name] = graph
    eviction = torch.empty(128*1024*1024,device="cuda",dtype=torch.uint8)
    samples = {name:[] for name in paths}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rng = random.Random(12)
    for _ in range(repeats):
        names = list(paths)
        rng.shuffle(names)
        for name in names:
            eviction.zero_()  # More than the RTX 5080's 64 MiB L2; outside timing.
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000)
    return {name:{"median_us":statistics.median(values),"samples_us":values}
            for name,values in samples.items()}


def projection_shapes(config):
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    d = config.get("head_dim", hidden // config["num_attention_heads"])
    q = config["num_attention_heads"] * d
    kv = config["num_key_value_heads"] * d
    return {"q_proj": (q, hidden), "kv_proj": (kv, hidden), "o_proj": (hidden, q),
            "qkv_fused": (q+2*kv, hidden), "gate_up_single": (intermediate, hidden),
            "gate_up_fused": (2*intermediate, hidden), "down_proj": (hidden, intermediate)}


@torch.inference_mode()
def run(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = {"gpu": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "timing_mode": "CUDA Graph GPU latency per call, repeated inputs/warm cache; no Python/quantization/transfers",
              "config": vars(args), "model_config": config, "results": []}
    if args.cold:
        report["timing_mode"] = "CUDA Graph GPU latency per call after 128 MiB L2 eviction; eviction outside timing"
    root = Path(__file__).resolve().parent
    sources = ["kernels/w4a16_small.cuh", "kernels/hgemm_pipeline.cuh", "kernels/hgemm_tma.cuh",
               "dandan_ops/csrc/hgemm.cu", "dandan_ops/csrc/hgemm_tma.cu"]
    report["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                               for name in sources if (root/name).is_file()}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    for projection, (n, k) in projection_shapes(config).items():
        if args.projections and projection not in args.projections:
            continue
        for m in args.ms:
            torch.manual_seed(m+n+k)
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            w = torch.randn(n, k, device="cuda", dtype=torch.float16)
            layer = ops.QuantizedLinear.from_float(w)
            p, s, z = layer.packed_weight, layer.scales, layer.zeros
            dense_ref = (x.float() @ w.float().t()).half()
            quant_ref = (x.float() @ layer.dequantize().half().float().t()).half()
            paths = {
                "torch_fp16": lambda: x @ w.t(),
                "hgemm_v8": lambda: ops.gemm(x, w, weight_transposed=True, backend="cuda"),
                "dense_auto": lambda: ops.gemm(x, w, weight_transposed=True),
                "w4a16_tensorcore": lambda: ops.quantized_linear(x, p, s, z, in_features=k, backend="tensorcore"),
                "w4a16_auto": lambda: layer(x),
            }
            if m == 1:
                paths["w4a16_gemv"] = lambda: ops.quantized_linear(x, p, s, z, in_features=k, backend="gemv")
            if "small" in args.extra and m <= 8:
                paths["w4a16_small"] = lambda: ops.quantized_linear(x, p, s, z, in_features=k, backend="small")
            if m % 128 == 0 and n % 64 == 0 and k % 64 == 0:
                paths["hgemm_tma"] = lambda: ops.gemm(x, w, weight_transposed=True, backend="tma")
            errors = {}
            for name, fn in paths.items():
                actual = fn()
                expected = quant_ref if name.startswith("w4a16") else dense_ref
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
                errors[name] = {"max_abs": (actual.float()-expected.float()).abs().max().item(),
                               "relative_l2": ((actual.float()-expected.float()).norm()/expected.float().norm()).item()}
            timing = (measure_cold if args.cold else measure)(paths, args.warmup, args.repeats, args.batch)
            report["results"].append({"projection": projection, "shape_MNK": [m,n,k],
                "layout": "A contiguous [M,K], weight contiguous [N,K]",
                "timing": timing, "errors": errors})
            print(f"{projection} M={m}: " + ", ".join(
                f"{name}={value['median_us']:.2f}" for name,value in timing.items()), flush=True)
            Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "myvllm/models/Qwen3-0.6B/config.json"))
    p.add_argument("--ms", nargs="+", type=int, default=[1,2,4,8,32,128,512])
    p.add_argument("--projections", nargs="+")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--cold", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--json", default="bench_qwen3_gemm.json")
    args = p.parse_args()
    if min(*args.ms, args.warmup, args.repeats, args.batch) < 1:
        p.error("counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
