"""Dense FP16 attention GPU timing with CUDA Graph replay (warm cache).

Includes output/scratch GPU work and any layout copies. Excludes Python
dispatch, capture, CPU/GPU transfers, and KV-cache updates. Q/K/V are dense
contiguous inputs. This is not an end-to-end engine benchmark.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import dandan_ops
from bench_hgemm import measure


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
        q = torch.randn(args.B, args.Hq, Q, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(args.B, args.Hkv, K, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        mask = None
        is_causal = not args.noncausal
        if is_causal and Q == 1 and start == K-1:
            # A suffix decode query sees the entire cache; no mask is needed.
            is_causal = False
        elif is_causal and start:
            mask = torch.arange(K, device="cuda")[None, :] <= torch.arange(Q, device="cuda")[:, None] + start
            is_causal = False
        paths = {
            "custom_cuda": lambda: dandan_ops.flash_attention(q, k, v, causal=not args.noncausal,
                                        backend="cuda", query_start=start),
            "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                        is_causal=is_causal, enable_gqa=True),
        }
        actual, expected = paths["custom_cuda"](), paths["sdpa"]()
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        errors = {"max_abs": (actual.float()-expected.float()).abs().max().item(),
                  "relative_l2": ((actual.float()-expected.float()).norm()/expected.float().norm()).item()}
        timing = measure(paths, args.warmup, args.repeats, args.batch)
        result = {"shape_B_Hq_Hkv_Q_K_D": [args.B, args.Hq, args.Hkv, Q, K, 128],
                  "causal": not args.noncausal, "query_start": start,
                  "timing": timing, "errors_vs_sdpa": errors,
                  "sdpa_over_custom": timing["sdpa"]["median_us"]/timing["custom_cuda"]["median_us"]}
        print(f"Q={Q}, K={K}, start={start}: " + ", ".join(
            f"{name}={values['median_us']:.2f} us" for name, values in timing.items()), flush=True)
        rows.append(result)
    report = {"gpu": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "timing_mode": "CUDA Graph GPU latency per call; repeated inputs/warm cache",
              "config": vars(args), "results": rows}
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--Hq", type=int, default=16)
    p.add_argument("--Hkv", type=int, default=8)
    p.add_argument("--Q", type=int, default=128)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--shapes", nargs="+", help="QxK pairs, e.g. 128x128 1x4096")
    p.add_argument("--query-start", type=int)
    p.add_argument("--noncausal", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--json", default="bench_flash_inference.json")
    args = p.parse_args()
    if min(args.B, args.Hq, args.Hkv, args.warmup, args.repeats, args.batch) <= 0 or args.Hq % args.Hkv:
        p.error("dimensions/counts must be positive and Hq must be divisible by Hkv")
    run(args)


if __name__ == "__main__":
    main()
