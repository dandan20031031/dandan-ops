"""Build an exact-shape profile from benchmarks/bench_qwen3_gemm.py measurements.

Keep the incumbent unless a candidate is at least --min-gain faster. Runtime
matching also requires the same GPU model, SM, PyTorch/CUDA build and layout.
This script writes data only, and never benchmarks in an inference forward.
"""
import argparse
import json
from pathlib import Path


def select(report, min_gain):
    environment = {key: report[key] for key in ("gpu", "sm", "torch", "torch_cuda")}
    routes = {"dense": {}, "w4a16": {}}
    decisions = []
    for row in report["results"]:
        m, n, k = row["shape_MNK"]
        key = f"{m}x{n}x{k}"
        times = {name: item["median_us"] for name, item in row["timing"].items()}
        for family, candidates, incumbent in (
            ("dense", {"torch_fp16": "torch", "hgemm_v8": "cuda", "hgemm_tma": "tma"},
             "hgemm_tma" if m % 128 == 0 and n % 64 == 0 and k % 64 == 0 else "hgemm_v8"),
            ("w4a16", {"w4a16_small": "w4a16_small", "w4a16_gemv": "w4a16_gemv",
                        "w4a16_tensorcore": "w4a16_tensorcore"},
             "w4a16_gemv" if m == 1 else "w4a16_tensorcore"),
        ):
            available = {name: route for name, route in candidates.items() if name in times}
            if incumbent not in available:
                continue
            winner = min(available, key=lambda name: times[name])
            if times[winner] > times[incumbent] * (1 - min_gain):
                winner = incumbent
            routes[family][key] = available[winner]
            decisions.append({"family": family, "shape": key, "incumbent": incumbent,
                              "selected": winner, "selected_us": times[winner],
                              "incumbent_us": times[incumbent]})
    return {"dispatch_revision": 1, "environment": environment,
            "layout": "contiguous A[M,K], weight[N,K]; FP16; INT4 group_size=64",
            "min_gain": min_gain, "routes": routes, "decisions": decisions}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("report")
    p.add_argument("--output", required=True)
    p.add_argument("--min-gain", type=float, default=0.1)
    args = p.parse_args()
    if not 0 <= args.min_gain < 1:
        p.error("min-gain must be in [0,1)")
    profile = select(json.loads(Path(args.report).read_text(encoding="utf-8")), args.min_gain)
    profile["source_report"] = Path(args.report).name
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote {sum(map(len, profile['routes'].values()))} routes to {destination}")


if __name__ == "__main__":
    main()
