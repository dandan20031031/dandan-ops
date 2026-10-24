"""Real Qwen3 paired Graph generation: native versus buffered decode metadata.

Both paths use the current measured INT4 routes and the engine's existing
batched token D2H. Only decode metadata allocation/H2D preparation changes.
Graph capture, INT4 packing and diagnostic profiling are outside timing.
"""
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys

import torch

from bench_qwen3_graph import Case, cpu_profile, load_prebuilt_extension, summary
from dandan_ops.integrations.myvllm import EngineQuantLinear, install_int4_projections
from dandan_ops.integrations.myvllm_decode import install_buffered_decode
import dandan_ops._dispatch as dispatch


METRICS = ("batch_prefill_ms", "decode_ms", "decode_step_ms",
           "decode_tokens_per_second", "total_ms")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def write_report(path, report):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


class MetadataCase(Case):
    """Reuse only Case.generation; bind buffered inputs before any capture."""

    def __init__(self, model, tokenizer, prompts, tokens, buffered, block_size,
                 *, capture_warmup=3, sampling_backend="torch"):
        from mini_engine.model import NativeQwen3Runner

        # Case.__init__ captures immediately, so its initialization is unsuitable
        # for installing a new fixed-address slab before graph construction.
        self.graph, self.measured = True, True
        self.buffered = buffered
        self.prompts, self.tokens = prompts, tokens
        self.batch = len(prompts)
        self.width = math.ceil((max(map(len, prompts)) + tokens) / block_size)
        scratch_count = max(self.batch, self.width)
        capacity = max(64, self.batch * self.width + scratch_count + 8)
        self.runner = NativeQwen3Runner(model, tokenizer, num_blocks=capacity,
                                      block_size=block_size, sampling_backend=sampling_backend)
        # All prompts fit in the first runtime.step(), so Case.generation's
        # prefill/decode boundary is the same for both cases.
        budget = max(4096, sum(map(len, prompts)), self.batch)
        self.runtime = self.runner.create_batch_runtime(
            use_cuda_graph=True, graph_batch_sizes=(self.batch,),
            graph_block_buckets=(self.width,), enable_prefix_cache=False,
            prefill_chunk_size=None, max_num_batched_tokens=budget)
        self.runtime.eos_token_ids.clear()
        self.scratch = tuple(self.runtime.graph_padding_block_ids)
        self.executor = None
        if buffered:
            self.executor = install_buffered_decode(self.runtime)
            self.bucket = self.executor.prepare(self.batch, self.width, warmup=capture_warmup)
        else:
            self.bucket = self.runtime.graph_manager.capture(
                self.batch, self.width, padding_block_ids=self.runtime.graph_padding_block_ids,
                warmup=capture_warmup)
        self.routes = []
        device = model.model.embed_tokens.weight.device
        for name, mod in model.named_modules():
            if isinstance(mod, EngineQuantLinear) and name.startswith("model.layers.0."):
                x = torch.empty(self.batch, mod.in_features, device=device, dtype=torch.float16)
                self.routes.append({"name": name, "n": mod.out_features, "k": mod.in_features,
                                    **mod.op.execution_info(x)})
        if not self.routes:
            raise AssertionError("no packed EngineQuantLinear projections were found")
        self.generations = 0
        self.last_cleanup = self.assert_clean()

    @property
    def name(self):
        return "graph_buffered_metadata" if self.buffered else "graph_original_metadata"

    def assert_clean(self):
        runtime, manager = self.runtime, self.runner.block_manager
        assert not runtime.scheduler.waiting and not runtime.scheduler.running, "unfinished requests"
        assert not runtime.enable_prefix_cache and len(runtime.prefix_cache) == 0, "prefix caching changed work"
        assert set(manager._owned) == {"__graph_padding__"}, "request KV ownership leaked"
        assert manager.table("__graph_padding__").block_ids == self.scratch, "scratch pages changed"
        assert manager.used_blocks == len(self.scratch), "only graph scratch pages should remain allocated"
        assert all(manager.refcount(page) == (1 if page in self.scratch else 0)
                   for page in range(manager.num_blocks)), "unexpected KV page references"
        return {"waiting": 0, "running": 0, "prefix_entries": 0,
                "used_blocks": manager.used_blocks, "scratch_blocks": len(self.scratch),
                "free_blocks": manager.free_blocks, "only_scratch_retained": True}

    def generation(self):
        self.assert_clean()
        metrics, tokens = super().generation()
        # Case.generation has already stopped its timer and synchronized.
        self.last_cleanup = self.assert_clean()
        self.generations += 1
        return metrics, tokens

    def execution_info(self):
        info = {"mode": self.name, "generations": self.generations,
                "batch": self.batch, "max_blocks": self.width,
                "capacity_blocks": self.runner.block_manager.num_blocks,
                "scratch_pages": list(self.scratch), "prefix_cache": False,
                "sampling_backend": self.runtime.sampling_backend,
                "token_d2h": "runtime._sample(...).cpu().tolist(); already batched in baseline",
                "bucket_inputs": {"tokens": self.bucket.input_ids.data_ptr(),
                                  "positions": self.bucket.positions.data_ptr(),
                                  "tables": self.bucket.block_tables.data_ptr()}}
        if self.executor is not None:
            info["buffered_executor"] = self.executor.execution_info()
        return info

    def close(self):
        if self.executor is not None:
            self.executor.close()
        for request in list(self.runtime.scheduler.waiting) + list(self.runtime.scheduler.running.values()):
            self.runtime.scheduler.abort(request.request_id)
        self.runtime.prefix_cache.clear()
        self.runner.block_manager.release("__graph_padding__")
        assert self.runner.block_manager.used_blocks == 0


@torch.inference_mode()
def run(args):
    os.environ["USE_TF"] = "0"
    engine = Path(args.engine).resolve()
    sys.path.insert(0, str(engine))
    engine_sources = ["mini_engine/engine/runtime.py", "mini_engine/engine/cuda_graph.py",
                      "mini_engine/engine/scheduler.py", "mini_engine/model/native_runner.py",
                      "mini_engine/model/native_qwen3.py", "mini_engine/sampling.py",
                      "mini_engine/cache/block_manager.py"]
    paths = {name: engine / name for name in engine_sources}
    decode_module = importlib.import_module("dandan_ops.integrations.myvllm_decode")
    paths["library/myvllm_decode.py"] = Path(decode_module.__file__)
    paths["benchmark/bench_decode_buffers.py"] = Path(__file__).resolve()
    paths["benchmark/bench_qwen3_graph.py"] = Path(sys.modules[Case.__module__].__file__)
    if Path(dispatch._path).is_file():
        paths["library/dispatch_profile.json"] = Path(dispatch._path)
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    profile_hash = object_sha256(dispatch._profile)
    if not dispatch._profile.get("routes", {}).get("w4a16"):
        raise RuntimeError("a current measured W4A16 profile is required")

    from mini_engine.config import QuantConfig
    from mini_engine.model import NativeQwen3Runner

    binary = load_prebuilt_extension(engine)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    loader = NativeQwen3Runner.from_pretrained(
        args.model, dtype=torch.float16, prefill_backend="auto", decode_backend="cuda",
        elementwise_backend="cuda", data_backend="cuda", linear_backend="torch",
        num_blocks=16, block_size=args.block_size, local_files_only=True,
        sampling_backend=args.sampling_backend)
    quantization = install_int4_projections(
        loader.model, QuantConfig("int4", activation_dtype="fp16", group_size=128))
    signature = dispatch._device_signature(torch.cuda.current_device())
    if dispatch._profile.get("environment") != signature:
        raise RuntimeError("measured dispatch profile does not match the current GPU/PyTorch/CUDA environment")
    package = Path(importlib.import_module("dandan_ops").__file__).resolve().parent
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    extensions = sorted(package.glob(f"_C.{abi}-*.pyd"))
    report = {
        "gpu": torch.cuda.get_device_name(), "sm": list(torch.cuda.get_device_capability()),
        "python": sys.version, "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "config": vars(args), "quantization": quantization,
        "engine_extension": str(binary), "engine_extension_sha256": sha256(binary),
        "dandan_extension_sha256": {str(path): sha256(path) for path in extensions},
        "runtime_py_sha256": source_hashes["mini_engine/engine/runtime.py"],
        "source_sha256_before": source_hashes, "dispatch_profile_sha256": profile_hash,
        "weights": "same immutable quantized model instance; independent runner, KV cache, scheduler and graphs",
        "timing_boundaries": {
            "generation": "Case.generation wall time: first runtime.step is prefill; following steps include scheduler, metadata preparation/H2D, graph replay, KV writes, greedy sampling and batched token D2H; fixed budget, EOS ignored",
            "decode_step_ms": "decode_ms divided by tokens-1; batch decode step latency, not per-request TPOT or GPU-only kernel time",
            "baseline": "current NativeBatchRuntime already uses _sample(...).cpu().tolist() for one batched token D2H",
            "change": "decode metadata allocations and H2D preparation only; same measured INT4 routes, sampler and scheduler",
            "excluded": "model loading/INT4 packing, graph capture, warmup, cleanup assertions, JSON and optional CPU profiling",
        },
        "results": [],
    }
    if not extensions:
        raise RuntimeError("cannot fingerprint the loaded dandan CUDA extension")

    def check_sources():
        after = {name: sha256(path) for name, path in paths.items()}
        report["source_sha256_after"] = after
        report["dispatch_profile_sha256_after"] = object_sha256(dispatch._profile)
        report["sources_unchanged"] = after == source_hashes and report["dispatch_profile_sha256_after"] == profile_hash
        if not report["sources_unchanged"]:
            report["invalid_reason"] = "source files or active dispatch profile changed during the benchmark"
            write_report(args.json, report)
            raise RuntimeError(report["invalid_reason"])

    for batch in args.batches:
        prompts = []
        for row in range(batch):
            text = args.prompt + " 请简洁回答。" * (row % 3)
            ids = loader.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            if len(ids) + args.tokens - 1 > loader.model.config.max_position_embeddings:
                raise ValueError("prompt and fixed output budget exceed the model context")
            prompts.append(ids)
        cases = []
        try:
            for buffered in (False, True):
                cases.append(MetadataCase(
                    loader.model, loader.tokenizer, prompts, args.tokens, buffered, args.block_size,
                    capture_warmup=args.capture_warmup, sampling_backend=args.sampling_backend))
            assert cases[0].runner is not cases[1].runner
            assert cases[0].runner.cache is not cases[1].runner.cache
            assert cases[0].runtime is not cases[1].runtime
            assert cases[0].runner.model is cases[1].runner.model
            assert cases[0].routes == cases[1].routes
            assert cases[0].width == cases[1].width
            row = {"batch": batch, "prompt_lengths": list(map(len, prompts)),
                   "table_width": cases[0].width, "routes": {case.name: case.routes for case in cases},
                   "execution_info": {"after_prepare": {case.name: case.execution_info() for case in cases}}}
            expected_tokens = None
            for warmup in range(args.warmup):
                for case in cases:
                    _, ids = case.generation()
                    if expected_tokens is None:
                        expected_tokens = ids
                    elif ids != expected_tokens:
                        raise AssertionError(f"warmup tokens differ for {case.name}, round {warmup}")
            row["execution_info"]["before_timing"] = {case.name: case.execution_info() for case in cases}
            samples = {case.name: [] for case in cases}
            token_checks = []
            rng = random.Random(args.seed + batch)
            for repeat in range(args.repeats):
                order = list(cases)
                rng.shuffle(order)
                paired = {}
                for order_index, case in enumerate(order):
                    metrics, ids = case.generation()
                    if ids != expected_tokens:
                        raise AssertionError(f"unstable tokens: {case.name}, round {repeat}")
                    sample = {"round": repeat, "order_index": order_index, **metrics,
                              "token_ids": ids, "token_sha256": object_sha256(ids),
                              "cleanup": dict(case.last_cleanup)}
                    samples[case.name].append(sample)
                    paired[case.name] = ids
                    print(f"B={batch} round={repeat} {case.name}: "
                          f"{metrics['decode_step_ms']:.4f} ms/decode step", flush=True)
                if paired[cases[0].name] != paired[cases[1].name]:
                    raise AssertionError(f"paired tokens changed in round {repeat}")
                token_checks.append({"round": repeat, "order": [case.name for case in order],
                                     "identical": True, "token_sha256": object_sha256(paired[cases[0].name])})
            row["generation"] = {
                name: {"samples": values, **{key: summary([sample[key] for sample in values])
                       for key in METRICS}} for name, values in samples.items()}
            row["paired_token_checks"] = token_checks
            row["all_rounds_tokens_identical"] = True
            row["execution_info"]["after_timing"] = {case.name: case.execution_info() for case in cases}
            buffered_name = "graph_buffered_metadata"
            previous = row["execution_info"]["before_timing"][buffered_name]["buffered_executor"]
            current = row["execution_info"]["after_timing"][buffered_name]["buffered_executor"]
            expected_steps = args.repeats * (args.tokens - 1)
            assert current["steps"] - previous["steps"] == expected_steps
            assert current["metadata_uploads"] - previous["metadata_uploads"] == expected_steps
            row["timed_buffered_steps"] = expected_steps
            row["timed_metadata_uploads"] = current["metadata_uploads"] - previous["metadata_uploads"]
            row["median_decode_step_speedup"] = (
                row["generation"]["graph_original_metadata"]["decode_step_ms"]["median"] /
                row["generation"][buffered_name]["decode_step_ms"]["median"])
            if args.profile_cpu:
                # Diagnostic generations are separate from all formal samples.
                row["cpu_profile_diagnostic_only"] = {case.name: cpu_profile(case.generation) for case in cases}
                row["execution_info"]["after_diagnostics"] = {case.name: case.execution_info() for case in cases}
            report["results"].append(row)
            check_sources()
            write_report(args.json, report)
            print(f"B={batch}: median decode step speedup "
                  f"{row['median_decode_step_speedup']:.3f}x; tokens identical", flush=True)
        finally:
            for case in cases:
                case.close()
            cases.clear()
            # Loop variables and the shuffled order also retain runner caches.
            case = None
            order = None
            gc.collect()
            torch.cuda.empty_cache()
    check_sources()
    write_report(args.json, report)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--engine", default=str(root / "myvllm"))
    parser.add_argument("--model", default=str(root / "myvllm/models/Qwen3-0.6B"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tokens", type=int, choices=(16, 32), default=16)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sampling-backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--prompt", default="用一句话解释 KV Cache。")
    parser.add_argument("--profile-cpu", action="store_true")
    parser.add_argument("--json", default="benchmarks/results/bench_decode_buffers.json")
    args = parser.parse_args()
    if min(*args.batches, args.repeats, args.warmup, args.capture_warmup, args.block_size) < 1:
        parser.error("batch sizes and counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
