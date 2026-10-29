# 基准与复现

运行库只读取 `dandan_ops/profiles/` 的离线路由；不会导入 benchmark，也不会运行时调优。
新报告默认写入忽略的 `build/benchmarks/`，避免每次测试在源码目录留下文件。
`benchmarks/results/` 只保留六份支撑当前结论的数据：

| 报告 | 保留原因 |
|---|---|
| [bench_qwen3_calibration_complete.json](../benchmarks/results/bench_qwen3_calibration_complete.json) | 当前 98 条 dense/W4A16 路由的校准来源 |
| [bench_elementwise.json](../benchmarks/results/bench_elementwise.json) | FP16/BF16 小算子结果，Graph 与 eager 分开 |
| [bench_flash_dispatch.json](../benchmarks/results/bench_flash_dispatch.json) | 自写 Attention、SDPA math、cuDNN 的对照 |
| [bench_qwen3_graph.json](../benchmarks/results/bench_qwen3_graph.json) | 旧/新路由的真实 Qwen3 Graph 验证 |
| [bench_decode_buffers.json](../benchmarks/results/bench_decode_buffers.json) | 旧引擎 16-token 元数据缓冲对照 |
| [bench_decode_buffers_32.json](../benchmarks/results/bench_decode_buffers_32.json) | 同条件 32-token 复测 |

历史迭代、失败/冒烟报告和旧开发日志已移出工作目录。先前已发布的报告可从
[清理前版本](https://github.com/dandan20031031/dandan-ops/tree/1800a2c/benchmarks/results)
查阅；本机额外保存完整 ZIP 快照，不作为安装依赖或随源码上传。
报告中的 source hash 对应测量时的代码；
本轮文档与目录整理不代表新的性能测量，也不改写旧数值或哈希。

## 可重复使用的入口

```powershell
python -m benchmarks.bench_hgemm --help
python -m benchmarks.bench_qwen3_gemm --config ../myvllm/models/Qwen3-0.6B/config.json --extra small
python scripts/select_gemm_routes.py build/benchmarks/bench_qwen3_gemm.json --output dandan_ops/profiles/qwen3_sm120_cu128.json
python -m benchmarks.bench_flash_inference --shapes 128x128 1x1024
python -m benchmarks.bench_flash_inference --D 256 --dtype bfloat16 --Hkv 4 --shapes 128x128 1x1024
python -m benchmarks.bench_attention_layouts --D 256 --dtype float16
python -m benchmarks.bench_elementwise --shapes 1x1024 8x1024 32x4096
```

HGEMM 提供独立矩阵形状对照及共用计时函数；Qwen3 GEMM 按真实配置推导投影形状，用于离线校准。
原先独立的早期 W4A16 入口已归档，其功能由前两者覆盖。
只有设备、GPU 架构、PyTorch/CUDA、布局和精确形状匹配时才使用 profile。
应在目标机器重新校准后更新；不要把选择 PyTorch 路由写成自写 GEMM 超过 cuBLAS。

以下可选入口还需要兼容引擎及自行取得的模型权重，脚本不修改引擎源码或触发引擎 JIT：

```powershell
python -m benchmarks.bench_qwen3_engine --help
python -m benchmarks.bench_qwen3_graph --engine ../myvllm --model ../myvllm/models/Qwen3-0.6B --batches 1 2 4 8 --tokens 16 --repeats 11
python -m benchmarks.bench_decode_buffers --engine ../myvllm --model ../myvllm/models/Qwen3-0.6B --batches 1 2 4 8 --tokens 16 --repeats 11 --warmup 2
```

decode 缓冲实验对应旧引擎 `ae967dd`，不是新版原生缓冲/图内采样实现的性能结论。
两路分别拥有 KV/cache/runtime/Graph；固定生成长度并忽略 EOS，包含调度、采样、token 回传和 KV 写入，
排除 prefill、capture、权重打包。不同 token 预算、不同 CUDA 版本的结果不能混算倍率。

## 当前小算子与 Attention 结果

2026-10-25，RTX 5080/SM120、Windows、PyTorch 2.9.1+cu128、CUDA 12.8。
FP16 `[8,1024]`，11 轮打乱顺序、16 calls/Graph，GPU 中位数 μs：

| 算子 | 自写 CUDA | 未融合 PyTorch |
|---|---:|---:|
| RMSNorm | 2.13 | 9.71 |
| 残差 + RMSNorm | 2.51 | 10.64 |
| SiLU × Mul | 1.47 | 2.35 |

这是重复输入/warm cache 的 Graph GPU 时间；普通 eager 摊销 wall time 在 JSON 中单独记录。
计时包含正常输出和布局转换，排除 capture、校验、H2D、模型和调度。
这些实现复用引擎已有融合算法，不能将相对未融合 PyTorch 的倍率当作新的端到端收益。

Attention 的 B=1/Hq=16/Hkv=8/D=128、FP16 causal 对照，Graph GPU 中位数 μs：

| Q/K | 自写 CUDA | 自动 SDPA（math） | 强制 cuDNN |
|---|---:|---:|---:|
| 128/128 | 9.98 | 40.94 | 7.54 |
| 1/1024，start=1023 | 24.29 | 53.34 | 5.75 |

本机 Flash SDPA 未编译；cuDNN 在两组测试中更快，不能只用 math 基线宣称超过成熟 Attention 实现。
benchmark 在计时外用 profiler 确认自动后端、探测强制后端；可用替代路径分别验证和计时。
暂未仅凭这两组形状修改运行时 Attention 选路。

2026-10-26 的 Attention 扩展增加 D64/128/256、FP16/BF16、滑窗、softcap、packed prefill 和
直接分页 decode。新入口 `bench_attention_layouts` 对比 packed kernel 与逐序列 SDPA，以及
直接分页 kernel 与 gather 后 SDPA；后者包含 gather GPU 工作。以上旧 D128 数据保持历史含义，
不能作为新接口或完整模型的性能结论。新测量默认写入忽略的 `build/benchmarks/`，包含源码哈希。

## 验证范围

回归测试包含 CPU/reference、数值契约、未对齐/尾块、stream、CUDA Graph、量化与可选引擎集成。
清理后的完整回归为 393 passed、1 skipped，跳过项为新版引擎不适用的旧 Graph API 模块。
Attention 扩展后的完整回归为 **528 passed、1 skipped**；新增测试和 sanitizer 范围见
[Attention 验证](ATTENTION.md#接入范围与复现)。
小型 Qwen3 FP16/BF16 prefill 和三步缓存 decode 的 logits 在容差内、argmax 一致。
GPU wheel 在独立安装目标及新进程中验证；复现命令见 [贡献说明](../CONTRIBUTING.md)。
可选引擎辅助函数优先检查分版本构建目录中的扩展，并核对当前 bindings 的导出接口；
接口缺失或 Python ABI 不兼容的旧二进制不会绑定到引擎。无兼容构建时明确报错，不触发 JIT。

小算子 memcheck 为 59 passed、0 errors；推理 racecheck 为 53 passed、0 errors、0 warnings。
包含梯度回退时，Compute Sanitizer 13.3 对 PyTorch 2.9.1/cu128 的
`layer_norm_grad_input_kernel_vectorized` 报告两条告警，与本库推理 kernel 分开记录。
原始日志已包含在清理前 ZIP 的 `build/` 中。

```powershell
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/test_elementwise.py -q -k "not cpu"
compute-sanitizer --tool racecheck --error-exitcode 99 python -m pytest tests/test_elementwise.py -q -k "not cpu and not autograd"
```
