"""Dense FP16/BF16 attention GPU timing with CUDA Graph replay (warm cache).

Includes output/scratch GPU work and any layout copies. Excludes Python
dispatch, capture, CPU/GPU transfers, and KV-cache updates. Q/K/V are dense
contiguous inputs. This is not an end-to-end engine benchmark.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import dandan_ops
if __package__:
    from .bench_hgemm import measure
else:
    from bench_hgemm import measure


BACKENDS = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}


def forced_sdpa(fn, name):
    def call():
        with sdpa_kernel(BACKENDS[name]):
            return fn()
    return call


def inspect_sdpa(fn):
    availability = {}
    for name, backend in BACKENDS.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with sdpa_kernel(backend):
                    fn()
                torch.cuda.synchronize()
                availability[name] = {"available": True}
            except RuntimeError as exc:
                availability[name] = {"available": False, "reason": str(exc)}
            availability[name]["warnings"] = [str(w.message) for w in caught]
    # CPU profiler operator names identify dispatch without timing profiling.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        fn()
        torch.cuda.synchronize()
    operators = sorted({e.key for e in profile.key_averages() if "_scaled_dot_product" in e.key})
    selected = [name for name in BACKENDS if any(
        {"math": "attention_math", "flash": "flash_attention", "efficient": "efficient_attention",
         "cudnn": "cudnn_attention"}[name] in op for op in operators)]
    return {"auto_backend": selected[0] if len(selected) == 1 else "unknown",
            "observed_operators": operators, "forced_backends": availability,
            "flash_compiled": torch.backends.cuda.is_flash_attention_available()}


@torch.inference_mode()
def run(args):
    torch.manual_seed(19)
    rows = []
    shapes = args.shapes or [f"{args.Q}x{args.K}"]
    for shape in shapes:
        Q, K = map(int, shape.lower().split("x"))
        start = args.query_start if args.query_start is not None else (K-Q if Q == 1 else 0)
        if min(Q, K) <= 0 or start < 0 or start + Q > K:
            raise ValueError("invalid Q/K/query_start")
        dtype = getattr(torch, args.dtype)
        tolerance = 1.6e-2 if dtype == torch.bfloat16 else 2e-3
        q = torch.randn(args.B, args.Hq, Q, args.D, device="cuda", dtype=dtype)
        k = torch.randn(args.B, args.Hkv, K, args.D, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        mask = None
        is_causal = not args.noncausal
        if is_causal and Q == 1 and start == K-1:
            # A suffix decode query sees the entire cache; no mask is needed.
            is_causal = False
        elif is_causal and start:
            mask = torch.arange(K, device="cuda")[None, :] <= torch.arange(Q, device="cuda")[:, None] + start
            is_causal = False
        if args.window_size is not None:
            if args.noncausal:
                raise ValueError("--window-size requires causal attention")
            pos = torch.arange(Q, device="cuda")[:, None] + start
            keys = torch.arange(K, device="cuda")[None, :]
            mask = (keys <= pos) & (keys >= pos - args.window_size + 1)
            is_causal = False
        paths = {
            "custom_cuda": lambda: dandan_ops.flash_attention(q, k, v, causal=not args.noncausal,
                                        backend="cuda", query_start=start, window_size=args.window_size),
            "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                        is_causal=is_causal, enable_gqa=True),
        }
        baseline = inspect_sdpa(paths["sdpa"])
        for name, status in baseline["forced_backends"].items():
            if status["available"] and name != baseline["auto_backend"]:
                paths[f"sdpa_{name}"] = forced_sdpa(paths["sdpa"], name)
        actual, expected = paths["custom_cuda"](), paths["sdpa"]()
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        for name, path in paths.items():
            if name.startswith("sdpa_"):
                torch.testing.assert_close(path(), expected, atol=tolerance, rtol=tolerance)
        errors = {"max_abs": (actual.float()-expected.float()).abs().max().item(),
                  "relative_l2": ((actual.float()-expected.float()).norm()/expected.float().norm()).item()}
        timing = measure(paths, args.warmup, args.repeats, args.batch)
        result = {"shape_B_Hq_Hkv_Q_K_D": [args.B, args.Hq, args.Hkv, Q, K, args.D],
                  "dtype": args.dtype, "window_size": args.window_size,
                  "causal": not args.noncausal, "query_start": start,
                  "sdpa_baseline": baseline,
                  "timing": timing, "errors_vs_sdpa": errors,
                  "sdpa_over_custom": timing["sdpa"]["median_us"]/timing["custom_cuda"]["median_us"]}
        print(f"Q={Q}, K={K}, start={start}: " + ", ".join(
            f"{name}={values['median_us']:.2f} us" for name, values in timing.items()), flush=True)
        rows.append(result)
    report = {"gpu": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "timing_mode": "CUDA Graph GPU latency per call; repeated inputs/warm cache",
              "config": vars(args), "results": rows}
    root = Path(__file__).resolve().parents[1]
    sources = ["dandan_ops/attention.py", "dandan_ops/csrc/flash_inference.cu",
               "kernels/attention_decode.cuh", "kernels/flash_tensorcore_inference.cuh"]
    report["source_sha256"] = {s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--Hq", type=int, default=16)
    p.add_argument("--Hkv", type=int, default=8)
    p.add_argument("--Q", type=int, default=128)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--D", type=int, choices=[64,128,256], default=128)
    p.add_argument("--dtype", choices=["float16","bfloat16"], default="float16")
    p.add_argument("--window-size", type=int)
    p.add_argument("--shapes", nargs="+", help="QxK pairs, e.g. 128x128 1x4096")
    p.add_argument("--query-start", type=int)
    p.add_argument("--noncausal", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--json", default="build/benchmarks/bench_flash_inference.json")
    args = p.parse_args()
    if min(args.B, args.Hq, args.Hkv, args.warmup, args.repeats, args.batch) <= 0 or args.Hq % args.Hkv:
        p.error("dimensions/counts must be positive and Hq must be divisible by Hkv")
    run(args)


if __name__ == "__main__":
    main()
