# Benchmarks

本目录集中保存 benchmark 脚本；`results/` 保存历史原始报告、校准与冒烟记录。
推理库位于 `dandan_ops/`，不依赖 benchmark 模块。

从仓库根目录运行，推荐使用模块入口：

```powershell
python -m benchmarks.bench_hgemm --help
python -m benchmarks.bench_qwen3_gemm --extra small
python -m benchmarks.bench_flash_inference --shapes 128x128 1x4096
python -m benchmarks.bench_decode_buffers --engine ../myvllm --model ../myvllm/models/Qwen3-0.6B
```

也支持 `python benchmarks/bench_hgemm.py ...` 的直接脚本入口。
自动保存报告的脚本默认写入 `benchmarks/results/`；`bench_w4a16.py` 默认仅输出到终端，
需要保存时指定 `--json benchmarks/results/bench_w4a16_local.json`。

真实模型脚本需要兼容引擎和独立取得的模型文件；计时边界、版本要求及复现命令见
[基准说明](../docs/BENCHMARKS.md)，历史报告分类见 [results/README.md](results/README.md)。
