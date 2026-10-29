# dandan_ops

[![CPU tests](https://github.com/dandan20031031/dandan-ops/actions/workflows/tests.yml/badge.svg)](https://github.com/dandan20031031/dandan-ops/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

面向模型推理的 PyTorch CUDA 算子库：FP16 GEMM、W4A16 INT4 Linear、GQA Attention，
以及 Norm/Softmax。将实验 kernel 封装为可安装、可调用、可验证的算子，保留显式后端选择、
数值约定和可复核的性能数据。

**当前实机验证：RTX 5080 / SM120，Windows，Python 3.13.9，PyTorch 2.9.1+cu128，
CUDA Toolkit 12.8，Visual Studio 2022 MSVC 14.44。** Linux 和其他 GPU 尚未实机验证。
自定义 CUDA 路径面向 forward inference；支持条件以各 API 为准。

## 功能

| 功能 | 实现与边界 |
|---|---|
| FP16 GEMM | `mma.sync` / `ldmatrix` / `cp.async`，尾块屏蔽；原生 Linear `[N,K]` 权重布局 |
| TMA GEMM | TMA + mbarrier 流水线，FP32 累加、FP16 直接写回；当前验证 SM120 |
| W4A16 Linear | INT4 packed 权重，group size 64，FP16 激活；GEMV、小 batch 和 Tensor Core 路径 |
| Attention | FP16/BF16、D64/128/256、GQA；dense、packed 变长 prefill、直接读取分页 KV 的 decode；causal/滑窗/softcap |
| Norm / Softmax | FP16/BF16/FP32 RMSNorm，FP32 LayerNorm/Softmax；梯度需求回退 PyTorch |
| 融合小算子 | FP16/BF16 残差 + RMSNorm、SiLU × Mul；明确舍入约定，不修改输入 |
| 后端选择 | `auto` + 显式后端；98 条离线路由严格匹配设备、版本和精确形状 |
| 可选引擎适配 | myvllm INT4 格式转换、投影替换、CUDA Graph decode 元数据缓冲复用 |

Attention 新增独立分页 decode 接口，读取物理 KV 页；KV 分配、写入和请求归属继续由引擎负责。
变长 prefill 使用 packed 布局。三种接口与 Graph metadata 更新约定见 [Attention 接入说明](docs/ATTENTION.md)。
当前尚未将新接口替换进 myvllm，也未验证 Qwen3.5 完整模型生成。

## 安装

```bash
git clone https://github.com/dandan20031031/dandan-ops.git
cd dandan-ops
```

扩展在安装时编译，import 不触发 JIT。
PyTorch 的 CUDA 版本需与本地 Toolkit 匹配；当前验证组合为 cu128 / CUDA 12.8。

### Windows / RTX 5080

安装 Python、CUDA Toolkit 12.8 和 VS2022 C++ Build Tools 后，在 PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
python -m pip install -e '.[dev]' --no-build-isolation
python -m pytest -q
```

`setup.py` 会初始化 VS2022 x64 环境。可用 `DANDAN_CUDA_HOME` 指定 Toolkit，
用 `DANDAN_CUDA_DRIVER_LIB` 指定包含 `cuda.lib` 的目录；默认会查找已安装 Toolkit。
当前 TMA 源码构建按 SM120 验证，其他架构需要单独编译与验证。
已有本地 myvllm 工具链时，也可运行 `scripts/build_for_myvllm.ps1 -EngineRoot ../myvllm`。

### CPU / reference 开发

无需 CUDA 扩展即可使用 CPU fallback 和量化格式接口。在装好 PyTorch、setuptools、wheel 后：

```bash
DANDAN_BUILD_CUDA=0 python -m pip install -e '.[dev]' --no-build-isolation
python -m pytest -q
```

PowerShell 使用 `$env:DANDAN_BUILD_CUDA='0'` 设置同一开关。GPU 专项和缺少 myvllm
checkout 的可选集成测试会跳过。CPU CI 不代表 CUDA 构建或性能已经跨平台验证。

## 使用

```python
import torch
import dandan_ops as ops

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(8, 1024, device=device, dtype=torch.float16)
weight = torch.randn(2048, 1024, device=device, dtype=torch.float16)

# Linear 原生布局：x=[M,K]，weight=[N,K]，输出=[M,N]。
y = ops.gemm(x, weight, weight_transposed=True, backend='auto')
print(ops.gemm_execution_info(x, weight, weight_transposed=True))

# 一次量化、保留 packed 权重；forward 不缓存完整反量化权重。
linear = ops.QuantizedLinear.from_float(weight, group_size=64)
y_int4 = linear(x)
print(linear.execution_info(x))
```

W4A16 保存 unsigned INT4（int32 内低 nibble 优先）、每组 scale/zero。
FP16 路径先把反量化权重舍入到 FP16，再做 FP32 累加；不同 kernel 的归约顺序可以不同，
不承诺逐位相等。量化损失与 kernel 计算误差在测试中分开衡量。

```python
q = torch.randn(1, 16, 128, 128, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 128, 128, device='cuda', dtype=torch.float16)
v = torch.randn_like(k)
out = ops.flash_attention(q, k, v, causal=True, backend='auto')

# suffix/decode 的绝对起点：query i 可见 key <= query_start+i。
decode = ops.flash_attention(q[:, :, :1].contiguous(), k, v,
                             causal=True, query_start=127)
```

`auto` 按支持条件和匹配的离线 profile 选择；显式 `cuda` 遇到不支持的输入会报错。
性能表不匹配当前 GPU/PyTorch/CUDA 时不会套用旧设备的路由。
各 API 的 shape、dtype、stride、scale 和 fallback 规则见 [完整技术说明](docs/DEVELOPMENT.md)。

Norm/Softmax 和融合小算子支持同样的 `auto/cuda/torch` 后端选择：

```python
weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
print(ops.elementwise_execution_info("rms_norm", x, weight))
normalized, summed = ops.residual_rms_norm(x, torch.zeros_like(x), weight)
activated = ops.silu_and_mul(normalized, summed)
```

低精度 RMSNorm 用 FP32 归约，LayerNorm 使用中心化方差，需梯度时回退 PyTorch。
执行信息会说明后端、回退原因及输入复制；Attention 通过 `flash_attention_execution_info`
查询。数值、空输入、布局和发布验证约定见 [小算子接口](docs/DEVELOPMENT.md#normsoftmax-与融合小算子)。

## 可选 myvllm 集成

库可独立安装；myvllm、模型权重和 tokenizer 不随仓库分发。集成测试通过 `MYVLLM_ROOT`
指向兼容的引擎 checkout，使用已经构建的引擎扩展，不触发引擎 JIT。
元数据缓冲适配器针对引擎 `ae967dd` 的旧 Graph API；新版引擎已经内置元数据缓冲和图内采样，
应直接使用其原生 runtime，库会拒绝重复安装旧版适配器，相应旧接口集成测试会跳过。

```python
from dandan_ops.integrations.myvllm import install_int4_projections
from dandan_ops.integrations.myvllm_decode import install_buffered_decode

# runner.model 已加载并完成 INT4 投影替换，Graph 尚未捕获。
runtime = runner.create_batch_runtime(use_cuda_graph=True)
executor = install_buffered_decode(runtime)
try:
    executor.prepare(4, 16)  # 启动阶段准备实际会使用的 bucket。
    # 后续照常 scheduler.add_request(...)、runtime.step()。
finally:
    executor.close()
```

INT4 适配器将引擎 signed INT4/group128 转为库的 affine unsigned INT4/group64，
转换保持反量化权重逐元素一致。缓冲适配器按 bucket 复用 pinned CPU/GPU slab，
每步一次 H2D 上传元数据，保留引擎采样与调度；要求单 CPU 执行线程、单 CUDA stream。
安装、scratch 所有权和关闭恢复约定见 [接入说明](docs/DEVELOPMENT.md#可选-decode-元数据缓冲适配)。

## 性能与验证

实测区分单算子、完整 forward 和实际生成循环，不把 microbenchmark 倍率当成端到端收益。
真实 Qwen3-0.6B、相同 INT4 后端、CUDA Graph 下，元数据缓冲复用的 decode step 延迟：

| Batch | 原生元数据 | 缓冲复用 |
|---:|---:|---:|
| 1 | 2.412 ms | 2.206 ms |
| 2 | 2.662 ms | 2.415 ms |
| 4 | 3.160 ms | 2.955 ms |
| 8 | 4.574 ms | 4.017 ms |

RTX 5080/cu128、固定生成 16 token、忽略 EOS、11 轮随机交错的中位数；每步处理 B 个请求。
包含调度、采样、token 回传和 KV 写入，排除 prefill、capture、权重打包。
32-token 预算复测延迟下降 6.5%–8.4%；配对轮次 token 一致。
这些是短上下文实验，不代表长上下文或在线服务的吞吐。

详见 [基准与复现](docs/BENCHMARKS.md)，支撑当前结论的原始 JSON 数据保留在仓库。
报告集中在 [benchmarks/results](benchmarks/results/)，供复核与离线校准使用；
运行时只读取 `dandan_ops/profiles/` 中的后端路由表。
历史发布验证：兼容 `ae967dd` 引擎的全部测试 **317 passed**；Norm/融合算子阶段新版引擎环境
**393 passed, 1 skipped**；Attention 扩展后为 **528 passed, 1 skipped**。
新增能力、135 项 Attention 测试及 sanitizer 范围见 [接入说明](docs/ATTENTION.md)。
跳过项为旧接口集成模块；ABI 发现测试覆盖 Windows/Linux 文件命名。
已有 CUDA 测试覆盖尾块、未对齐输入、非默认 stream、CUDA Graph、数值契约、KV 隔离和异常恢复。
本轮新增 Norm/Softmax 数值回归、梯度回退及 FP16/BF16 融合测试，也验证了小型 Qwen3
prefill 与三步缓存 decode。构建和 wheel 验证命令见 [贡献说明](CONTRIBUTING.md)。

## 文件保留原则

`dandan_ops/`、`kernels/`、安装配置和许可证是功能与构建所需文件。
`tests/` 是长期回归测试，负责防止后续修改重新引入数值、Graph 和内存边界问题；
`benchmarks/` 与 `scripts/` 用于校准和复现，运行库不导入它们。
源码安装包不包含测试、benchmark、历史报告或开发文档。

新测量默认输出到忽略的 `build/benchmarks/`。`build/`、缓存、编译临时文件和日志
可在进程退出后清理；本机正在使用的 `dandan_ops/_C` 扩展应保留，删除后需重新编译。
旧测量、旧文档和清理前的构建产物已保存在项目外的本地 ZIP 归档中。

## 常用算子的补充顺序

现有 GEMM、量化、Norm/融合和 dense/packed/paged Attention 已覆盖主要计算。
下一步优先评估 **RoPE → KV cache 写入 → greedy argmax**，可复用引擎已有算法，
围绕实际模型调用补齐接口。三者目前尚未作为本库公开算子提供；具体范围与验收见
[算子补充评估](docs/DEVELOPMENT.md#常用算子补充评估)。

## 项目结构

```text
dandan-ops/
├── dandan_ops/              Python 运行库
│   ├── csrc/                算子注册、校验与 CUDA launcher
│   ├── integrations/        可选引擎适配
│   ├── profiles/            运行时后端路由表
│   ├── functional.py        函数接口
│   ├── attention.py         dense / packed / paged Attention 与 metadata plan
│   ├── elementwise.py       Norm、Softmax 与融合小算子
│   ├── quantization.py      INT4 格式与量化
│   └── quantized_linear.py  W4A16 Linear
├── kernels/                 CUDA kernel 实现
├── benchmarks/              benchmark 脚本
│   └── results/             6 份支撑当前结论的原始报告
├── tests/                   数值、边界与可选引擎集成测试
├── scripts/                 构建、路由选择、wheel 验证
├── docs/                    接口约定与基准说明
├── .github/workflows/       CI
├── setup.py
├── pyproject.toml
└── LICENSE
```

从仓库根目录通过 `python -m benchmarks.bench_qwen3_gemm ...` 运行基准；
各入口见 [基准说明](docs/BENCHMARKS.md)。

欢迎带上 GPU、PyTorch/CUDA 版本、shape、后端与最小复现提交 Issue 或 Pull Request。
贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证与来源

[MIT License](LICENSE)。kernel 源自作者的 CUDA 实验并适配为独立库；来源文件、依赖和
未分发内容见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
