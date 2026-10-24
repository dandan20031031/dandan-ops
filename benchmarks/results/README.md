# Benchmark reports

这里保存原始测量与离线校准数据，供性能复核使用，不在模型推理时加载。
运行时路由表位于 `dandan_ops/profiles/qwen3_sm120_cu128.json`。

| 文件 | 用途 |
|---|---|
| `bench_decode_buffers*.json` | 实际生成循环中，原生/缓冲元数据的配对测量 |
| `bench_qwen3_graph.json` | eager/Graph、旧/新后端对照 |
| `bench_qwen3_engine*.json` | 单请求 eager 生成循环 |
| `bench_qwen3_before/after.json` | Qwen3 形状优化前后原始样本 |
| `bench_qwen3_calibration*.json` | 选择精确形状路由所用的校准记录 |
| `bench_qwen3_o_proj.json` | 补充 o_proj 投影形状 |
| `bench_qwen3_candidates.json` | 候选后端测量 |
| `bench_qwen3_cold.json` | 独立的冷 L2 实验 |
| `bench_flash*.json` | dense GQA Attention，含 suffix 测试 |
| `bench_hgemm.json` / `bench_w4a16.json` | 历史 cu130 数据 |
| `bench_qwen3_graph_smoke.json` | 冒烟运行记录，不作为正式性能结论 |

归档只改变文件位置，保留原始数值、token、元数据和哈希。
环境与计时边界见 [基准说明](../../docs/BENCHMARKS.md)。
