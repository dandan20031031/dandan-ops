import argparse
import json
import time
import torch
import dandan_ops


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def time_cuda(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return median(samples), samples


def run_case(m, n, k, warmup, repeats, device):
    torch.manual_seed(1234 + m + n + k)
    weight = torch.randn(n, k, device=device, dtype=torch.float16)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    layer = dandan_ops.QuantizedLinear.from_float(weight, group_size=64)

    dense = lambda: x @ weight.t()
    quantized = lambda: layer(x)
    dense_us, _ = time_cuda(dense, warmup, repeats)
    quant_us, _ = time_cuda(quantized, warmup, repeats)

    with torch.no_grad():
        actual = quantized()
        restored = dandan_ops.dequantize_int4(
            layer.packed_weight, layer.scales, layer.zeros, k, 64
        )
        # Tensor Core path rounds dequantized tiles to FP16 before MMA.
        if m != 1:
            restored = restored.half().float()
        reference = (x.float() @ restored.t()).to(torch.float16)
        max_abs = (actual - reference).abs().max().item()
        mean_abs = (actual - reference).abs().mean().item()

    fp16_bytes = weight.numel() * weight.element_size()
    int4_bytes = (
        layer.packed_weight.numel() * layer.packed_weight.element_size()
        + layer.scales.numel() * layer.scales.element_size()
        + layer.zeros.numel() * layer.zeros.element_size()
    )
    return {
        "M": m, "N": n, "K": k,
        "dense_fp16_us": dense_us,
        "w4a16_us": quant_us,
        "w4a16_over_dense": quant_us / dense_us,
        "dense_weight_mib": fp16_bytes / 2**20,
        "w4a16_state_mib": int4_bytes / 2**20,
        "memory_ratio": int4_bytes / fp16_bytes,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "reference_semantics": "FP32 dequant" if m == 1 else "FP32 dequant -> FP16 MMA operands",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    cases = [(1, 4096, 4096), (8, 4096, 4096), (32, 4096, 4096)]
    results = [run_case(*case, args.warmup, args.repeats, args.device) for case in cases]
    metadata = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    print(json.dumps(metadata, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
