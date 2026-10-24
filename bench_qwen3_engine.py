"""Compare packed projection dispatch with real local Qwen3 weights.

Single-request eager TPOT, same INT4 state and prompt for both routes. Uses
the engine's existing CUDA decode/elementwise/data module without rebuilding
or editing the engine. Timing includes Python/sampling/synchronization and
cache updates. This benchmark intentionally does not claim batching results.
"""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys

import torch
import dandan_ops._dispatch as dispatch
from dandan_ops.integrations.myvllm import EngineQuantLinear, install_int4_projections


@contextlib.contextmanager
def route_profile(enabled):
    saved = dispatch._profile
    if not enabled:
        dispatch._profile = {}
    try:
        yield
    finally:
        dispatch._profile = saved


@torch.inference_mode()
def run(args):
    os.environ["USE_TF"] = "0"
    sys.path.insert(0, str(Path(args.engine).resolve()))
    from mini_engine.config import QuantConfig
    import mini_engine.kernels.cuda_extension as extension
    from mini_engine.model import NativeQwen3Runner

    # Load exactly the current prebuilt engine extension. Fail if it is absent;
    # keep the comparison independent of compiler/JIT changes in that checkout.
    binary = Path(args.engine) / "build/torch_extensions/paged_attention/miniqwen_paged_attention.pyd"
    if not binary.is_file():
        raise FileNotFoundError(f"build engine extension before this benchmark: {binary}")
    spec = importlib.util.spec_from_file_location("miniqwen_paged_attention", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extension.load_cuda_extension = lambda *a, **kw: module

    runner = NativeQwen3Runner.from_pretrained(args.model, dtype=torch.float16,
        prefill_backend="auto", decode_backend="cuda", elementwise_backend="cuda",
        data_backend="cuda", linear_backend="torch", num_blocks=64, block_size=16,
        local_files_only=True)
    quant_report = install_int4_projections(runner.model,
        QuantConfig("int4", activation_dtype="fp16", group_size=128))
    layers = [mod for mod in runner.model.modules() if isinstance(mod, EngineQuantLinear)]
    # Inspect both actual model shapes and routes; no floating weight cache.
    routes = []
    for name, mod in runner.model.named_modules():
        if isinstance(mod, EngineQuantLinear) and name.startswith("model.layers.0."):
            x = torch.zeros(1, mod.in_features, device="cuda", dtype=torch.float16)
            routes.append({"name": name, "execution": mod.op.execution_info(x)})
    samples = {"previous_dispatch": [], "measured_dispatch": []}
    token_ids = {}
    for name in samples:
        with route_profile(name == "measured_dispatch"):
            for _ in range(args.warmup): runner.generate_incremental(args.prompt, args.tokens)
    rng = random.Random(37)
    for _ in range(args.repeats):
        names = list(samples)
        rng.shuffle(names)
        for name in names:
            with route_profile(name == "measured_dispatch"):
                result = runner.generate_incremental(args.prompt, args.tokens)
            if len(result.token_ids) < 2:
                raise RuntimeError("at least two generated tokens are required for TPOT")
            samples[name].append({"tpot_ms": result.tpot_ms, "ttft_ms": result.ttft_ms,
                                  "total_ms": result.total_latency_ms, "output_tokens": len(result.token_ids)})
            token_ids.setdefault(name, result.token_ids)
            if result.token_ids != token_ids[name]: raise AssertionError("unstable token sequence")
            print(name, samples[name][-1], flush=True)
    if token_ids["previous_dispatch"] != token_ids["measured_dispatch"]:
        raise AssertionError("changed greedy tokens between projection routes")
    # Compare logits under teacher-forced continuation, not only token IDs.
    ids = runner.tokenizer.apply_chat_template([{"role":"user", "content":args.prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0].cuda()
    caches = [runner.model.create_cache(64,16) for _ in range(2)]
    blocks = list(range((ids.numel()+args.tokens+15)//16))
    max_abs, relative_l2 = 0.0, 0.0
    for step in range(min(args.tokens, 8)):
        inputs = ids if step == 0 else torch.tensor([token_ids["previous_dispatch"][step-1]], device="cuda")
        position = 0 if step == 0 else ids.numel()+step-1
        with route_profile(False): before = runner.model(inputs,caches[0],blocks,position)[-1].float()
        with route_profile(True): after = runner.model(inputs,caches[1],blocks,position)[-1].float()
        torch.testing.assert_close(after,before,atol=0,rtol=0)
        max_abs = max(max_abs,(after-before).abs().max().item())
        relative_l2 = max(relative_l2, ((after-before).norm()/before.norm()).item())
    report = {"gpu":torch.cuda.get_device_name(), "torch":torch.__version__, "torch_cuda":torch.version.cuda,
        "config": vars(args), "timing_mode":"single-request eager wall time; includes Python, sampling, sync and KV updates",
        "quantization":quant_report, "first_layer_routes":routes, "samples":samples,
        "median_ms":{name:{key:statistics.median(row[key] for row in rows)
                          for key in ("tpot_ms","ttft_ms","total_ms")} for name,rows in samples.items()},
        "greedy_tokens_identical":True, "token_ids":token_ids,
        "logits_comparison":{"steps":min(args.tokens,8),"max_abs":max_abs,"relative_l2":relative_l2},
        "packed_layers":len(layers), "graph":False, "batch":1,
        "note":"same packed weights; only profile enabled/disabled. No comparison to original floating weights."}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report["median_ms"],indent=2),flush=True)


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--engine",default=str(root/"myvllm"))
    p.add_argument("--model",default=str(root/"myvllm/models/Qwen3-0.6B"))
    p.add_argument("--prompt",default="用一句话解释 KV Cache。")
    p.add_argument("--tokens",type=int,default=16)
    p.add_argument("--warmup",type=int,default=1)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--json",default="benchmarks/results/bench_qwen3_engine.json")
    args = p.parse_args()
    if min(args.tokens,args.warmup,args.repeats) < 1: p.error("counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
