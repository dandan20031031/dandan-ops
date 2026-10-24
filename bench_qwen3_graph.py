"""Real-weight Qwen3: paired eager/Graph and previous/measured INT4 dispatch.

Two timing boundaries are reported separately: fixed-context full-model
forward, and the actual engine scheduler loop (metadata, greedy sampling,
KV updates and token transfer included). Capture/packing are outside timing.
The generation workload ignores EOS and runs a fixed token budget per request.
Only this benchmark process uses the optional adapter; engine sources stay
unchanged. Previous and measured dispatch always own separate captured graphs.
"""
import argparse
import cProfile
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pstats
import random
import statistics
import sys
import time

import torch
from dandan_ops.integrations.myvllm import EngineQuantLinear, install_int4_projections
from bench_qwen3_engine import route_profile


def load_prebuilt_extension(engine):
    import mini_engine.kernels.cuda_extension as extension
    path = Path(engine) / "build/torch_extensions/paged_attention/miniqwen_paged_attention.pyd"
    if not path.is_file():
        raise FileNotFoundError(f"build engine CUDA extension first: {path}")
    name = "miniqwen_paged_attention"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    extension.load_cuda_extension = lambda *a, **kw: module
    return path


class DisabledPrefixCache:
    def lookup(self, tokens): return None
    def insert(self, *args, **kwargs): return None
    def clear(self): pass


def summary(values):
    ordered = sorted(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "p90": ordered[min(len(ordered)-1, math.ceil(0.9*len(ordered))-1)]}


def cpu_profile(fn):
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    torch.cuda.synchronize()
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({"file": file, "line": line, "name": name, "calls": calls,
                     "self_ms": own*1000, "cumulative_ms": cumulative*1000})
    return sorted(rows, key=lambda row: row["cumulative_ms"], reverse=True)[:30]


class Case:
    def __init__(self, model, tokenizer, prompts, tokens, graph, measured, block_size):
        from mini_engine.model import NativeQwen3Runner
        self.graph, self.measured = graph, measured
        self.prompts, self.tokens = prompts, tokens
        self.batch = len(prompts)
        self.width = math.ceil((max(map(len,prompts))+tokens) / block_size)
        scratch = max(self.batch,self.width)
        capacity = max(64,self.batch*self.width+scratch+8)
        self.runner = NativeQwen3Runner(model, tokenizer, num_blocks=capacity, block_size=block_size)
        self.runtime = self.runner.create_batch_runtime(use_cuda_graph=graph,
            graph_batch_sizes=(self.batch,), graph_block_buckets=(self.width,))
        self.runtime.prefix_cache = DisabledPrefixCache()
        self.runtime.eos_token_ids.clear()  # Equal work even when a prompt reaches EOS.
        self.bucket = None
        with route_profile(measured):
            if graph:
                self.bucket = self.runtime.graph_manager.capture(self.batch,self.width,
                    padding_block_ids=self.runtime.graph_padding_block_ids,warmup=3)
            self.routes = []
            for name, mod in model.named_modules():
                if isinstance(mod,EngineQuantLinear) and name.startswith("model.layers.0."):
                    x = torch.empty(self.batch,mod.in_features,device="cuda",dtype=torch.float16)
                    self.routes.append({"name":name,**mod.op.execution_info(x)})

    @property
    def name(self):
        return ("graph" if self.graph else "eager") + ("_measured" if self.measured else "_previous")

    def generation(self):
        from mini_engine.engine.request import GenerationRequest
        runtime = self.runtime
        requests = [GenerationRequest(f"bench-{row}",list(ids),max_new_tokens=self.tokens)
                    for row,ids in enumerate(self.prompts)]
        for request in requests: runtime.scheduler.add_request(request)
        torch.cuda.synchronize()
        start = time.perf_counter()
        runtime.step()
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        while runtime.scheduler.waiting or runtime.scheduler.running:
            runtime.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        generated = [r.output_token_ids for r in requests]
        assert all(len(ids)==self.tokens for ids in generated)
        return {"batch_prefill_ms":(prefill_end-start)*1000,
                "decode_ms":(end-prefill_end)*1000,
                "decode_step_ms":(end-prefill_end)*1000/(self.tokens-1),
                "decode_tokens_per_second":self.batch*(self.tokens-1)/(end-prefill_end),
                "total_ms":(end-start)*1000}, generated

    def seed_probe(self):
        manager = self.runner.block_manager
        tables, tokens = [], []
        with route_profile(self.measured):
            for row, ids in enumerate(self.prompts):
                blocks = manager.allocate(f"probe-{row}",self.width*manager.block_size)
                logits = self.runner.model(torch.tensor(ids,device="cuda"),self.runner.cache,blocks)
                tokens.append(logits[-1].argmax())
                tables.append(blocks)
        self.probe_tokens = torch.stack(tokens)
        self.probe_positions = torch.tensor(list(map(len,self.prompts)),device="cuda",dtype=torch.long)
        self.probe_tables = torch.tensor(tables,device="cuda",dtype=torch.long)
        self.initial_probe_tokens = self.probe_tokens.clone()
        self.initial_probe_positions = self.probe_positions.clone()

    def forward(self):
        if self.graph:
            return self.bucket.replay(self.probe_tokens,self.probe_positions,self.probe_tables)
        return self.runner.model.forward_decode_batch(self.probe_tokens,self.runner.cache,
                                                     self.probe_tables,self.probe_positions)

    def close_probe(self):
        for row in range(self.batch): self.runner.block_manager.release(f"probe-{row}")


@torch.inference_mode()
def validate_probes(cases, steps):
    metrics = {"steps":steps,"same_route_eager_graph_max_abs":{},"cross_route":[]}
    for step in range(steps):
        outputs = {}
        for case in cases:
            with route_profile(case.measured): outputs[case.name] = case.forward().clone()
        for route in ("previous","measured"):
            eager, graph = outputs[f"eager_{route}"],outputs[f"graph_{route}"]
            torch.testing.assert_close(graph,eager,atol=2e-3,rtol=2e-3)
            assert torch.equal(eager.argmax(-1),graph.argmax(-1))
            metrics["same_route_eager_graph_max_abs"][route] = max(
                metrics["same_route_eager_graph_max_abs"].get(route,0),(eager-graph).abs().max().item())
        before, after = outputs["eager_previous"].float(),outputs["eager_measured"].float()
        error = (after-before).norm()/before.norm().clamp_min(1e-12)
        assert torch.isfinite(after).all() and error < 0.01
        metrics["cross_route"].append({"max_abs":(after-before).abs().max().item(),
            "relative_l2":error.item(), "argmax_identical":bool(torch.equal(after.argmax(-1),before.argmax(-1)))})
        # Teacher force all cases with the same tokens to keep KV histories comparable.
        tokens = outputs["eager_previous"].argmax(-1)
        for case in cases:
            case.probe_tokens.copy_(tokens)
            case.probe_positions.add_(1)
    for case in cases:
        case.probe_tokens.copy_(case.initial_probe_tokens)
        case.probe_positions.copy_(case.initial_probe_positions)
    return metrics


@torch.inference_mode()
def measure_forward(cases,args):
    samples = {case.name:[] for case in cases}
    for case in cases:
        with route_profile(case.measured):
            for _ in range(args.warmup): case.forward()
        torch.cuda.synchronize()
    start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rng = random.Random(61)
    for _ in range(args.repeats):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            with route_profile(case.measured):
                torch.cuda.synchronize()
                wall = time.perf_counter()
                start.record()
                case.forward()
                end.record()
                end.synchronize()
                elapsed = (time.perf_counter()-wall)*1000
            samples[case.name].append({"wall_ms":elapsed,"event_ms":start.elapsed_time(end)})
    return {name:{"samples":rows, "wall_ms":summary([row["wall_ms"] for row in rows]),
                  "event_ms":summary([row["event_ms"] for row in rows])} for name,rows in samples.items()}


@torch.inference_mode()
def run(args):
    os.environ["USE_TF"] = "0"
    sys.path.insert(0,str(Path(args.engine).resolve()))
    from mini_engine.model import NativeQwen3Runner
    from mini_engine.config import QuantConfig
    binary = load_prebuilt_extension(args.engine)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    runner = NativeQwen3Runner.from_pretrained(args.model,dtype=torch.float16,
        prefill_backend="auto",decode_backend="cuda",elementwise_backend="cuda",
        data_backend="cuda",linear_backend="torch",num_blocks=16,block_size=args.block_size,
        local_files_only=True)
    quantization = install_int4_projections(runner.model,QuantConfig("int4",activation_dtype="fp16",group_size=128))
    report = {"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"torch_cuda":torch.version.cuda,
        "config":vars(args),"quantization":quantization,"engine_extension_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
        "timing_boundaries":{"forward":"same full table width and context per case; output allocation/KV write included; Graph also copies device metadata; CUDA Event includes launch gaps",
                             "generation":"actual engine scheduler, greedy sampling and per-token CPU transfer; fixed budget, EOS disabled; capture/packing excluded; prefill listed separately"},
        "results":[]}
    for batch in args.batches:
        prompts = []
        for row in range(batch):
            text = args.prompt + (" 请简洁回答。"*(row%3))
            ids = runner.tokenizer.apply_chat_template([{"role":"user","content":text}],
                tokenize=True,add_generation_prompt=True,enable_thinking=False)
            prompts.append(ids)
        cases = [Case(runner.model,runner.tokenizer,prompts,args.tokens,graph,measured,args.block_size)
                 for graph in (False,True) for measured in (False,True)]
        row = {"batch":batch,"prompt_lengths":list(map(len,prompts)),"table_width":cases[0].width,
               "routes":{case.name:case.routes for case in cases}}
        generated, samples = {}, {case.name:[] for case in cases}
        for case in cases:
            with route_profile(case.measured):
                for _ in range(args.warmup): case.generation()
        rng = random.Random(51)
        for _ in range(args.repeats):
            order = list(cases)
            rng.shuffle(order)
            for case in order:
                with route_profile(case.measured): sample,ids = case.generation()
                if case.name in generated: assert generated[case.name] == ids
                generated[case.name] = ids
                samples[case.name].append(sample)
        for route in ("previous","measured"):
            assert generated[f"eager_{route}"] == generated[f"graph_{route}"]
        row["generation"] = {name:{"samples":values, **{key:summary([v[key] for v in values])
            for key in values[0]}} for name,values in samples.items()}
        row["tokens"] = generated
        row["same_route_eager_graph_tokens_identical"] = True
        row["previous_measured_tokens_identical"] = generated["eager_previous"] == generated["eager_measured"]
        for case in cases: case.seed_probe()
        row["validation"] = validate_probes(cases,min(args.validate_steps,args.tokens-1))
        row["forward"] = measure_forward(cases,args)
        if args.profile_cpu:
            row["cpu_profile"] = {}
            for case in cases:
                with route_profile(case.measured): row["cpu_profile"][case.name] = cpu_profile(case.forward)
        for case in cases: case.close_probe()
        report["results"].append(row)
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"B={batch}: " + ", ".join(f"{name}={v['decode_step_ms']['median']:.3f} ms/step"
              for name,v in row["generation"].items()),flush=True)
        del cases, case
        gc.collect()
        torch.cuda.empty_cache()


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--engine",default=str(root/"myvllm"))
    p.add_argument("--model",default=str(root/"myvllm/models/Qwen3-0.6B"))
    p.add_argument("--batches",type=int,nargs="+",default=[1,2,4,8])
    p.add_argument("--tokens",type=int,default=16)
    p.add_argument("--repeats",type=int,default=7)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--block-size",type=int,default=16)
    p.add_argument("--validate-steps",type=int,default=8)
    p.add_argument("--prompt",default="用一句话解释 KV Cache。")
    p.add_argument("--profile-cpu",action="store_true")
    p.add_argument("--json",default="benchmarks/results/bench_qwen3_graph.json")
    args = p.parse_args()
    if min(*args.batches,args.warmup,args.repeats,args.validate_steps,args.block_size) < 1 or args.tokens < 2:
        p.error("counts must be positive, tokens>=2")
    run(args)


if __name__ == "__main__": main()
