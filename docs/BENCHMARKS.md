# Benchmark scope and reproduction

All saved results describe the recorded machine and software versions. They
are not performance guarantees. Historical cu130 data and current cu128 data
must not be combined into a speedup ratio.

Saved JSON reports live in [benchmarks/results](../benchmarks/results/).
Benchmark scripts write there by default and create missing output directories.
These reports are development evidence and calibration inputs; inference reads
only the packaged dispatch table in `dandan_ops/profiles/`.

## Operators

- `bench_hgemm.py/json`: historical FP16 HGEMM, TMA and W4A16 comparison.
- `bench_qwen3_gemm.py`: derives Qwen3 projection shapes from a local model
  config; synthetic inputs, CUDA Graph, FP32 oracles outside timing.
- `benchmarks/results/bench_qwen3_calibration_complete.json`: seven projection shapes, seven M
  values, 49 measured shapes. Profile contains 98 dense/W4A16 routes.
- `benchmarks/results/bench_qwen3_cold.json`: separate L2 eviction experiment, not mixed with
  warm-cache measurements.
- `bench_flash_inference.py/json`: dense D=128 FP16 GQA attention, not paged KV.

```powershell
python bench_qwen3_gemm.py --config ../myvllm/models/Qwen3-0.6B/config.json --extra small --json benchmarks/results/bench_local.json
python scripts/select_gemm_routes.py benchmarks/results/bench_local.json --output dandan_ops/profiles/qwen3_sm120_cu128.json
python bench_flash_inference.py --shapes 128x128 1024x1024 1x1024 1x4096
```

Run calibration on the target machine before replacing a profile. Routes
require matching GPU, SM, PyTorch, CUDA, layout and exact shape; forward does
not time kernels or autotune. A dense route selecting PyTorch/cuBLAS is a
dispatch improvement and does not establish that custom HGEMM beats cuBLAS.

## Real Qwen3 generation

These scripts require a compatible local myvllm checkout, its already-built
CUDA extension, and separately downloaded Qwen3-0.6B weights/tokenizer.
They load the engine extension without building it and only modify objects
inside the benchmark process.

The saved metadata-adapter measurements use the legacy Graph API, matching
engine commit `ae967dd`. Newer engine revisions have native metadata buffering
and in-Graph sampling. Use their native runtime instead of installing this
legacy adapter; it rejects that newer API. The saved results are historical
paired measurements of the original baseline, not a claim about newer engine
performance. Legacy integration tests accept a compatible checkout through
`MYVLLM_ROOT` and skip when the newer API is detected.

```powershell
python bench_qwen3_graph.py --engine ../myvllm --model ../myvllm/models/Qwen3-0.6B --batches 1 2 4 8 --tokens 16 --repeats 11 --json benchmarks/results/bench_graph_local.json
python bench_decode_buffers.py --engine ../myvllm --model ../myvllm/models/Qwen3-0.6B --batches 1 2 4 8 --tokens 16 --repeats 11 --warmup 2 --profile-cpu --json benchmarks/results/bench_buffers_local.json
python bench_decode_buffers.py --engine ../myvllm --model ../myvllm/models/Qwen3-0.6B --batches 1 2 4 8 --tokens 32 --repeats 11 --warmup 2 --json benchmarks/results/bench_buffers_32_local.json
```

`benchmarks/results/bench_qwen3_graph.json` compares previous/measured dispatch in separate eager
and Graph cases. Graphs are captured independently after selecting the route.
Reported Event times and actual generation-loop wall times have distinct
boundaries. Graph output views must be cloned before another replay if kept
for comparison.

`benchmarks/results/bench_decode_buffers.json` and `benchmarks/results/bench_decode_buffers_32.json` compare native
metadata versus buffered metadata using the **same measured INT4 routes**.
Each case owns independent cache, scheduler, runtime, scratch pages and Graph.
Both native and buffered paths already transfer sampled tokens in one batch.
The change is decode metadata allocation, preparation and upload.

The reports include shuffled samples, tokens, release checks and source/binary
hashes. Timing includes scheduler, metadata upload, Graph replay, KV writes,
sampling and token D2H. Loading, packing, capture, warmup, assertion/JSON work
and diagnostic profiling are outside timing. Prefill is listed separately.
EOS is disabled to preserve equal fixed work; this differs from natural-EOS
single-request TPOT results in `benchmarks/results/bench_qwen3_engine_long.json`.

Public report path strings use portable labels in place of absolute local
paths. Numeric samples, generated tokens and recorded hashes were preserved.
Historical source hashes identify the code at measurement time; later
packaging/documentation changes are not another timing run.

See [development notes](DEVELOPMENT.md) for full measurements, numerical
contracts and the supported integration boundary.
