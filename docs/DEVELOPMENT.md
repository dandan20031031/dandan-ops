# dandan_ops 历史开发记录

以下保存了各轮开发时的接口、验证与性能记录。当前安装方法、测试数量和引擎版本边界以
[项目首页](../README.md)为准；新版 myvllm 已内置元数据缓冲和图内采样，使用其原生 runtime。

从 `cuda_code` 独立出的推理算子库。原实验目录只作为来源，不参与新项目构建。

## 构建与验证

当前构建已对齐 `myvllm/.venv`：RTX 5080 (SM120)、Python 3.13.9、
PyTorch 2.9.1+cu128、CUDA Toolkit 12.8、Visual Studio 2022 MSVC 14.44。
`setup.py` 自动初始化完整的 VS2022 x64 编译环境，避免混入 VS18 的头文件/链接器，
不使用 `-allow-unsupported-compiler` 或调试模式 `-G`。

```powershell
cd dandan-ops
powershell -ExecutionPolicy Bypass -File scripts\build_for_myvllm.ps1 -EngineRoot ..\myvllm
..\myvllm\.venv\Scripts\python.exe -m pytest -q
```

该脚本使用引擎虚拟环境及其 `.cuda-12.8/Library` 工具链。CPU fallback 无需 CUDA 扩展。
扩展在安装时编译，import 不会 JIT 编译。当前构建产物只在本机 SM120 上验证；
跨架构的 TMA 编译尚未验证，不能将本机 pyd 直接用于其他 Python/PyTorch/GPU 环境。

## 普通 GEMM：复用原 HGEMM

```python
import torch
import dandan_ops as ops

x = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
weight = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

# 推荐 Linear 权重布局：weight=[N,K]，无需每次转置/拷贝
out = ops.gemm(x, weight, weight_transposed=True)
# 标准矩阵接口：b=[K,N]；自定义后端可能需要一次布局转换
out = ops.gemm(x, weight.t())

# 显式对照
v8 = ops.gemm(x, weight, weight_transposed=True, backend="cuda")
v10 = ops.gemm(x, weight, weight_transposed=True, backend="tma")
reference = ops.gemm(x, weight, weight_transposed=True, backend="torch")
```

返回 `[M,N]` FP16。`weight_transposed=True` 指传入 B 的转置，即 Linear 原生的 `[N,K]` 存储。
`backend="auto"`：CPU/其他 dtype/需要 autograd 时回退 PyTorch。CUDA FP16 推理优先匹配离线
性能表：GPU 型号、SM、PyTorch/CUDA 版本、精确 M/N/K，以及连续 `[M,K]` / `[N,K]` 布局。
匹配时可选择 PyTorch/cuBLAS、V8 或 TMA；未命中时保留原有规则：SM120 且
`M%128=0, N%64=0, K%64=0` 选 V10 TMA，其余选 V8 衍生流水线。
显式 `cuda/tma` 不支持当前配置时会报错，不静默换后端。

```python
print(ops.gemm_execution_info(x, weight, weight_transposed=True))
# implementation: torch / cuda / tma；reason 会说明离线 profile 或默认规则。
```

- V8 衍生版保留原 128B swizzle、ldmatrix fragment 映射、mma.sync.m16n8k16、
  FP32 累加和两级 cp.async。新增尾块屏蔽和 FP16 epilogue；`M<=32` 时 warp 沿 N 方向划分，
  改善 decode/小 prefill 的利用率。原版大 tile 是 128×64×64。
- V10 从原 `hgemm_v10_tma.cu` 提取，保留 TMA、mbarrier、L2 superblock 和 FP32 累加，
  现在 epilogue 直接写 FP16，去掉完整 FP32 输出缓冲区和独立转换 kernel。
- 当前没有 split-K、bias/activation epilogue 融合，也没有承诺超过 cuBLAS。

## Inference FlashAttention

```python
q = torch.randn(1, 16, 128, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
with torch.inference_mode():
    out = ops.flash_attention(q, k, v, causal=True, backend="cuda")

    # 连续 KV cache 的 decode：K/V 包含历史和当前 token
    q_decode = torch.randn(1, 16, 1, 128, device="cuda", dtype=torch.float16)
    out_decode = ops.flash_attention(q_decode, k, v, causal=True,
                                    query_start=k.shape[2]-1, backend="cuda")
```

当前 inference backend 支持：

```text
Q: [B, Hq, Q, 128]
K/V: [B, Hkv, K, 128]
Hq % Hkv == 0 (GQA)
FP16、causal、Q <= K、query_start、有限的自定义 scale
```

只实现前向推理，未实现 dropout/backward。Qwen3 所需的 GQA 不会展开 K/V 头数。

- `Q>1`：从旧 V2.1 移植 `ldmatrix`、`mma.sync.m16n8k16`、FP32 累加和 online softmax。
  tile 为 64×32，Q fragment 驻寄存器，K/V 双缓冲 `cp.async`，不分配完整 Q×K 分数矩阵。
  softmax 概率转 FP16 后参与 PV 的 Tensor Core 乘法，存在正常的 FP16 舍入误差。
- `Q=1`：专用 decode kernel，每 256 个 KV token 一段，warp 归约并在 FP32 合并 softmax
  状态；短 cache 一次 launch，长 cache 增加一次合并 launch。
- causal 规则：第 i 个 query 仅可见 `kv_position <= query_start+i`。
  普通 prefill 传 0；后缀 query 传 K-Q；单 token decode 传 K-1。
  不能将 SDPA 的矩形 `is_causal=True` 直接当作 decode mask，它默认从左上角对齐。
- `backend="cuda"` 强制自写 kernel，不支持时报告错误；`auto` 对其他 dtype、设备或
  autograd 需求回退 SDPA；`torch` 强制 SDPA。所有路径都保留 GQA、scale、query_start 语义。
  非连续或地址未对齐的输入会复制，拷贝成本需计入接入性能。

目前输入为**连续逻辑布局的 dense K/V**。myvllm 的 prefill 可在原 GQA 展开之前调用：

```python
# mini_engine/kernels/prefill.py 的接入位置；模型侧已经完成 RoPE。
from dandan_ops import flash_attention

# q=[B,Hq,Q,128]，k/v=[B,Hkv,Q,128]；直接返回同布局的 attention 输出
output = flash_attention(q, k, v, causal=True, backend="cuda")
```

myvllm 的 decode 使用 `[num_blocks, block_size, Hkv, D]` 分页 cache，
还需要独立的 paged-attention kernel，读取 block_tables/context_lens 并直接定位物理块。
本次未修改引擎文件，也未将分页 cache 展开复制后冒充分页 kernel。
下一步应沿用引擎的分页接口实现 decode，再做完整模型的 logits/token 和延迟验证。

性能验证命令：

```powershell
..\myvllm\.venv\Scripts\python.exe bench_flash_inference.py --shapes 128x128 1024x1024 1x1024 1x4096
..\myvllm\.venv\Scripts\python.exe bench_flash_inference.py --Q 128 --K 1024 --query-start 896 --json benchmarks/results/bench_flash_suffix.json
```

首轮 RTX 5080 / PyTorch 2.9.1+cu128 验证数据（B=1、Hq=16、Hkv=8、D=128，单位 μs）：

| 场景 | 自写 CUDA | PyTorch SDPA |
|---|---:|---:|
| causal prefill Q=K=128 | 8.45 | 40.03 |
| causal prefill Q=K=1024 | 91.48 | 1327.20 |
| suffix Q=128, K=1024, start=896 | 38.41 | 140.50 |
| decode Q=1, K=128, start=127 | 12.59 | 21.51 |
| decode Q=1, K=1024, start=1023 | 24.28 | 53.53 |
| decode Q=1, K=4096, start=4095 | 28.38 | 317.87 |

原始样本和误差见 `benchmarks/results/bench_flash_inference.json`、`benchmarks/results/bench_flash_suffix.json`。CUDA Graph GPU 计时，重复输入、warm cache，
包含输出/scratch 对应的 GPU 工作，不含 Python 调用、图捕获、KV cache 写入或端到端调度。
本机 PyTorch 报告 `is_flash_attention_available() == False`，该 SDPA 基线不代表官方
FlashAttention 或其他平台最快 attention 实现。Q<K 且 start=0 的 causal 场景只访问前 Q 个 KV，
不能据此评估长历史 cache 的吞吐。分页 decode 和多卡尚未验证。

FlashAttention 专项覆盖 57 项测试（FP32 SDPA oracle、边界形状、GQA、suffix mask、
自定义 scale、非连续/未对齐输入、CUDA Graph、非默认 stream 和回退语义），
Compute Sanitizer memcheck 为 0 errors，racecheck 为 0 hazards。
额外通过小型 D=128 Qwen3 模型测试：FP16 / INT4 两种投影分别替换测试进程中的 prefill，
对照原 math prefill 的 prompt logits 与三步分页 decode，logits 在容差内且 argmax 一致。
此测试不等于真实权重完整模型验证，也不修改引擎源码。

## INT4 QuantizedLinear

```python
layer = ops.QuantizedLinear.from_float(weight, group_size=64)
# 已测 Qwen3 小 batch -> small；未命中 profile 时保留 GEMV/Tensor Core 默认规则
single = layer(x[:1])
batch = layer(x)

# 同一个量化状态可对比后端
layer.backend = "naive"       # 旧朴素 CUDA GEMM（仅作对照）
old = layer(x)
layer.backend = "tensorcore"  # 强制新融合路径，含 M=1
new = layer(x)
layer.backend = "small"       # 强制 M<=8 的 warp 归约路径
small = layer(x[:8])
layer.backend = "reference"   # FP32 反量化 + matmul
ref = layer(x)
layer.backend = "auto"
```

量化方案：沿 K 按 64 个元素分组、包含零的 min/max 仿射量化、`q=round(w/scale+zero)`，
限制到 `[0,15]`。每个 int32 从低 nibble 开始存 8 个 unsigned INT4。

```text
packed_weight: [N, ceil(K/8)] int32
scales:        [N, ceil(K/group_size)] float32
zeros:         [N, ceil(K/group_size)] int32
```

底层接口：

```python
q, scales, zeros = ops.quantize_int4(weight, group_size=64)
packed = ops.pack_int4(q)
unpacked = ops.unpack_int4(packed, original_k=weight.shape[1])
restored = ops.dequantize_int4(packed, scales, zeros, weight.shape[1], 64)
out = ops.quantized_linear(x, packed, scales, zeros,
                          in_features=weight.shape[1], group_size=64)
```

融合 W4A16 在 kernel 内解包当前 tile、用 FP32 scale/zero 反量化并转 FP16，写入原 swizzle
shared 布局，随后执行原 ldmatrix/MMA 核心。只保留 packed 权重，不创建完整的 FP16 权重矩阵。
激活通过 cp.async 加载，量化权重使用寄存器解包后写入双缓冲 shared memory。
CUDA 快速路径要求 FP16 激活、group_size=64；CPU/FP32/BF16/其他分组走 reference。
新增 `small` kernel：一个 warp 负责一个输出列，lane 按连续 packed word 读取，
复用每次解包的权重至 M<=8 个激活行；8 lane 共享 scale/zero，激活支持 16B 向量加载，
只做 warp shuffle 归约。尾块及未对齐输入有屏蔽/标量路径。
M=1 保留原 256-thread GEMV 的加法归约树；已测输入与旧 GEMV 逐位一致，避免改变
单请求模型 decode 的舍入结果。M=2..8 使用跨行复用的独立归约路径，以 FP32 oracle 对拍。
`auto` 和严格 `cuda` 均可依据实测表在 packed kernels 间选择，`cuda` 不允许回退 reference。
显式 `gemv`、`small`、`tensorcore`、`naive` 保留强制路径；`execution_info(x)` 可查看实际调用。

**数值约定**：当前所有 FP16 W4A16 路径（small、GEMV、Tensor Core、naive、reference）
均在乘法前把反量化权重舍入到 FP16，再以 FP32 乘加。归约顺序可以不同，不承诺逐位相等。
BF16/FP32 输入的 reference 使用 FP32 反量化权重。
量化损失（相对原始权重）和 kernel 计算误差（相对相同量化状态）也应分开评估。
Bias 保持现有行为：乘法结果先转输入 dtype，再做 bias 加法。

## Qwen3 真实投影形状：cu128 优化

上一轮从本地 Qwen3-0.6B config 推导六种 N/K，分别测试 M=1/2/4/8/32/128/512，共 42 个形状。
本轮补齐每层必用的 o_proj `[N=1024,K=2048]`，总计七种 N/K、49 个形状和 98 条路由。
完整校准表为 `benchmarks/results/bench_qwen3_calibration_complete.json`，新增原始样本见 `benchmarks/results/bench_qwen3_o_proj.json`。
输入为合成张量，量化/FP32 oracle 在计时外。前后均为 RTX 5080、PyTorch 2.9.1+cu128，
CUDA Graph、warm cache、8 次调用/graph、21 轮打乱顺序。单位 μs。

| 投影 / M,N,K | 原 W4A16 auto | 新 W4A16 auto |
|---|---:|---:|
| Q / 1,2048,1024 | 9.65 | 3.84 |
| Q / 2,2048,1024 | 20.96 | 4.10 |
| Q / 8,2048,1024 | 20.92 | 9.81 |
| Gate 或 Up / 1,3072,1024 | 13.24 | 4.54 |
| Down / 2,1024,3072 | 61.98 | 8.40 |
| Down / 8,1024,3072 | 62.00 | 23.28 |

原始样本：`benchmarks/results/bench_qwen3_before.json`、`benchmarks/results/bench_qwen3_after.json`；离线选择依据为
`benchmarks/results/bench_qwen3_calibration.json`。后端选择至少要求 10% 的实测优势，否则保留原默认后端。
这是离线精确形状选择，forward 不做计时、调优或读写磁盘，不按未测 M 插值。
大多数被测 dense GEMM 会选 PyTorch/cuBLAS；这是调度收益，不是自写 HGEMM 加速。
M>=32 的 W4A16 Tensor Core 路径本轮没有结构性提速，仍需要进一步优化。
TMA FP16 直写已减少缓冲和 launch，但其计算主循环仍可能主导延迟，不能保证每个形状提速。
另外保存 `benchmarks/results/bench_qwen3_cold.json`：每次调用前用 128 MiB buffer 清空 L2，清空过程不计时，
M=1/2/4/8 的新 small 路径仍有收益；该数据不用于与 warm-cache 表混算加速比。

```powershell
..\myvllm\.venv\Scripts\python.exe bench_qwen3_gemm.py --extra small --json benchmarks/results/bench_qwen3_local.json
..\myvllm\.venv\Scripts\python.exe scripts\select_gemm_routes.py benchmarks/results/bench_qwen3_local.json --output dandan_ops\profiles\qwen3_sm120_cu128.json
# 新进程导入后读取新 profile；换 GPU、PyTorch/CUDA 时不会沿用不匹配的表。
..\myvllm\.venv\Scripts\python.exe bench_qwen3_gemm.py --ms 1 2 4 8 --extra small --cold --batch 1 --json benchmarks/results/bench_qwen3_cold.json
..\myvllm\.venv\Scripts\python.exe bench_qwen3_engine.py --warmup 3 --repeats 21 --tokens 32 --json benchmarks/results/bench_qwen3_engine_long.json
```

`bench_qwen3_engine.py` 使用本地真实权重、相同 INT4 状态对比旧默认调度与新 profile，
只在基准进程中装入适配器，不编辑引擎文件。它要求已构建的引擎 CUDA 扩展，
不触发 JIT；测量单请求 eager TPOT，包含 Python、采样、同步和 KV 更新，未使用 CUDA Graph。

真实权重结果见 `benchmarks/results/bench_qwen3_engine_long.json`：旧默认调度 TPOT 中位数 17.50 ms，
新调度 17.66 ms（21 轮，均生成 17 token 后 EOS）。该轮未观察到稳定的 TPOT 提升；
不能把单算子的倍率当成整模型倍率。两种路径生成 token 一致，检查的 8 步 logits
逐位一致（max_abs=0）。下一轮应分析 eager 发射/调度开销并在引擎 CUDA Graph 下复测。

## 真实模型 CUDA Graph 对照

`bench_qwen3_graph.py` 复用引擎现有 `DecodeGraphManager`，在同一份真实 Qwen3-0.6B 权重和
INT4 状态上对比旧默认 / 新实测后端。两种后端各自捕获独立图：捕获后切换 Python profile
不会改变图中的 kernel。引擎源码不参与本轮修改，加载既有扩展，不触发 JIT。

RTX 5080、PyTorch 2.9.1+cu128、FP16 激活、无 projection fusion，11 轮随机交错测量。
以下为**实际引擎生成循环的 decode ms/step 中位数**，同一步并行处理 B 个请求；
不包含 prefill、打包或 capture，包含 CPU 调度、元数据准备、采样、token 回传和 KV 写入。

| B | eager 旧后端 | eager 新后端 | Graph 旧后端 | Graph 新后端 | Graph 内旧/新 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.386 | 10.743 | 3.576 | 2.536 | 1.41× |
| 2 | 13.761 | 11.205 | 11.859 | 2.800 | 4.24× |
| 4 | 14.258 | 11.071 | 12.008 | 3.638 | 3.30× |
| 8 | 14.319 | 10.718 | 12.373 | 4.884 | 2.53× |

每个请求固定生成 16 token，忽略 EOS，以保持相同计算预算；不同请求使用 18/23/28 长度
的 prompt。本轮 Graph bucket 的 table width=3，分别记录完整 forward 和生成循环的计时边界。
生成循环沿用引擎真实的动态 table width 规则；受控 forward 对照则使用相同 table width
和 context。前述单请求自然 EOS 数据是历史基线，不能与此处的固定预算结果混算倍率。

结果说明：小 batch 自写 kernel 的收益在 Graph 模式下变成了模型级收益，B=1 的 eager
仍未稳定提速。固定上下文 forward 的新 Graph CUDA Event 中位数分别为
2.029 / 2.233 / 3.035 / 3.968 ms（B=1/2/4/8）；这部分不包含采样和 CPU 调度。
eager 的 Event 时间包含 host 发射造成的 GPU 间隙，不能把它当作纯 kernel 时间。

诊断用 cProfile 显示，eager 每个 decode step 有 196 次投影调用和相应的元数据检查/路由；
Graph replay 消除了这些重复 Python 工作。诊断 profile 不用于正式计时，累计耗时不能
直接相加。当前引擎已采用批量 token D2H；下面的新适配器继续减少 decode 元数据分配和 H2D 拷贝。

正确性：全部测试 prompt 的生成 token 一致；同一后端的 eager/Graph 在检查的 8 步
logits 上逐位一致。跨新旧 kernel，B=1 仍逐位一致；B>1 的归约顺序不同，最大相对 L2
为 0.280%，最大绝对 logit 差为 0.09082，单独报告在 JSON 中，不声称逐位相同。
新增小模型测试覆盖 B=1/2/4/8 的 FP16/INT4、不同长度、非连续物理页、全 batch→部分→全
batch、scratch 与 live KV 隔离，并检查 inactive KV 不变。

```powershell
..\myvllm\.venv\Scripts\python.exe bench_qwen3_graph.py --batches 1 2 4 8 --tokens 16 --repeats 11 --warmup 2 --validate-steps 8 --profile-cpu --json benchmarks/results/bench_qwen3_graph.json
```

原始样本、CPU 调用统计、实际路由和误差都在 `benchmarks/results/bench_qwen3_graph.json`；这是本机短上下文
固定 batch 的实验结果，不代表更长上下文、混合到达负载或完整服务的吞吐。

## 可选 decode 元数据缓冲适配

`dandan_ops.integrations.myvllm_decode.install_buffered_decode(runtime)` 可安装到当前
myvllm 的 `NativeBatchRuntime` 实例，无需修改引擎源码。权重量化和后端选择完成后、
Graph 捕获前安装；保持模型权重、scratch 配置和执行 stream 固定。
它只替换该实例的 `decode_step`，prefill、调度、prefix cache、采样参数和 token 回传沿用引擎。
默认检查 runner 的 `__graph_padding__` allocator owner 持有这些页且引用计数为 1；
如通过其他 owner 预留 scratch，可显式传入 `scratch_owner=...`，这些页不得属于活跃请求或 prefix cache。

每个 Graph bucket 分配一个 pinned CPU slab 和一个固定 GPU slab，连续保存
`tokens[B] / positions[B] / block_tables[B,W]`；Graph 输入直接引用 GPU slab。
每步按调度后的当前请求顺序重填全部行和尾列，一次异步 H2D 后 replay，不再逐行创建 CUDA tensor。
inactive slot 使用各自私有 scratch 写页，允许共享只读 prefix，但当前写页必须独占。
CPU 完整校验先于上传和惰性 capture；采样异常会先等待已提交 GPU 工作完成，再由引擎回收请求页。

该适配器需要 NumPy，wheel 提供 `myvllm` extra。原 myvllm 环境已安装此依赖。

```python
from dandan_ops.integrations.myvllm_decode import install_buffered_decode

# runner 的模型/权重/量化后端已就绪；缓存需能容纳 live pages 和 Graph scratch。
runtime = runner.create_batch_runtime(
    use_cuda_graph=True,
    graph_batch_sizes=(1, 2, 4, 8),
    graph_block_buckets=(4, 8, 16),
)
executor = install_buffered_decode(runtime)
try:
    # 启动时 prepare 将会使用的 bucket；未准备的 bucket 会在首次使用时捕获。
    executor.prepare(4, 8)
    # 后续照常 scheduler.add_request(...)、runtime.step()。
    print(executor.execution_info())  # pinned 状态、地址、上传计数
finally:
    executor.close()  # 等待 GPU，恢复该实例原 decode_step；已有 Graph 仍可用。
```

当前实现面向单 CPU 执行线程、单 CUDA stream、同步采样回传的 Graph decode；安装到已捕获的 runtime 会拒绝。
每个 bucket 的 pinned/GPU slab 各占 `8*B*(2+W)` 字节。`prepare()` 可在启动阶段完成
分配和 capture，正式计时只包含 replay 路径。更换权重、stream 或 scratch 时应创建新 runtime。

`bench_decode_buffers.py` 在同一份真实 Qwen3-0.6B INT4 权重上，用独立 KV/cache/runtime/Graph
对比当前原生元数据路径和缓冲路径，双方使用相同的实测 GEMM routes。
原生路径已经执行一次批量 token D2H；本轮收益仅归因于元数据准备、分配和上传路径变化。
两路均关闭 prefix cache、greedy、忽略 EOS，固定生成 16 token；包含调度、采样、KV 写入
和 token 回传，排除 prefill、capture、权重打包；正式测量外另存 CPU profile。

RTX 5080、PyTorch 2.9.1+cu128、FP16 激活、INT4 权重，每组 2 次预热和 11 轮随机交错测量。
单位为 batch decode **ms/step 中位数**，同一步处理 B 个请求；两列预算分别计时，不能跨预算混算倍率。

| B | 16-token 原生 | 16-token 缓冲 | 32-token 原生 | 32-token 缓冲 |
|---:|---:|---:|---:|---:|
| 1 | 2.4120 | 2.2057 | 2.4794 | 2.3171 |
| 2 | 2.6615 | 2.4153 | 2.7383 | 2.5179 |
| 4 | 3.1597 | 2.9551 | 3.2678 | 3.0550 |
| 8 | 4.5744 | 4.0171 | 4.4882 | 4.1133 |

16-token 预算延迟下降 6.5%–12.2%，32-token 复测下降 6.5%–8.4%；prompt 长度为 18/23/28。
两个预算各自采用相同 table width（分别为 3、4），所有配对轮次的 token 一致。
16/32-token 每个 B 的正式 decode 分别为 165/341 步，上传计数也分别为 165/341，
全部 slab 地址保持不变。两次运行均验证源码/profile hash 未变化、每轮结束仅保留 scratch 页。
原始数据分别在 `benchmarks/results/bench_decode_buffers.json` 和 `benchmarks/results/bench_decode_buffers_32.json`。

```powershell
..\myvllm\.venv\Scripts\python.exe bench_decode_buffers.py --batches 1 2 4 8 --tokens 16 --repeats 11 --warmup 2 --profile-cpu --json benchmarks/results/bench_decode_buffers.json
..\myvllm\.venv\Scripts\python.exe bench_decode_buffers.py --batches 1 2 4 8 --tokens 32 --repeats 11 --warmup 2 --json benchmarks/results/bench_decode_buffers_32.json
```

JSON 保存逐轮样本、token 一致性、元数据上传计数、固定地址、KV 释放检查、环境和源码 hash。
这仍是本机短上下文实验；长上下文和混合到达服务吞吐需要各自复测。

## 历史性能对照（cu130，保留记录）

RTX 5080、cu130、N=K=4096、group_size=64。单位 μs，越小越快。
`bench_hgemm.py` 使用 CUDA Graph replay，4 次调用/graph，5 次 warmup、15 轮打乱测量顺序。
原始样本和逐路径误差见 `benchmarks/results/bench_hgemm.json`。FP16 HGEMM 列使用已是 `[N,K]` 的权重。

| M | PyTorch FP16 | V8 HGEMM | V10 TMA | 旧 W4A16 naive | 新 W4A16 auto | 旧/新 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.00 | 34.00 | — | 359.02 | 39.47 | 9.1× |
| 8 | 19.58 | 31.16 | — | 889.22 | 93.75 | 9.5× |
| 32 | 30.92 | 34.72 | — | 3144.85 | 62.71 | 50.1× |
| 128 | 51.74 | 85.89 | 57.74 | 11590.86 | 113.62 | 102.0× |
| 512 | 176.98 | 241.92 | 204.87 | 45343.97 | 245.46 | 184.7× |

对于 N=K=4096，量化权重（含 scale/zero）为 10 MiB，原 FP16 权重为 32 MiB。
这表示权重存储节省 68.75%，不等于整模型显存节省相同比例。
该历史版本量化 GEMM 明显快于旧 naive，但仍慢于 FP16 cuBLAS；不要与 cu128 新数据混算加速比。

**计时边界**：图重放下的每次调用 GPU 延迟，包含图内输出分配对应的 GPU 工作、布局转换（仅
`hgemm_v8_input_KN_with_copy` 路径）和 V10 FP32→FP16 转换；不包含 Python 调用开销、
权重量化、图捕获、CPU/GPU 传输。重复输入属于 warm-cache 场景，不代表端到端服务。
旧 `benchmarks/results/bench_w4a16.json` 使用不同的 CUDA Event 单次计时，作为历史基线保留，不能混算加速比。
尤其对连续的普通 `[K,N]` 权重，每次拷贝布局代价很大，建议预先存为 `[N,K]`。

```powershell
python bench_hgemm.py --warmup 5 --repeats 15 --batch 4 --json benchmarks/results/bench_hgemm.json
```

## 其他现有 API

| API | 当前实现/约束 |
|---|---|
| `softmax(x)` | CUDA FP32 `[M,N]`, N%4=0；当前为共享内存归约版本（已非原两遍 V5） |
| `rms_norm(x, weight, eps)` | RMSNorm V3，CUDA FP32 `[M,D]`, D%4=0 |
| `layer_norm(x, weight, bias, eps)` | LayerNorm V2，CUDA FP32 `[M,D]` |
| `batch_norm_infer(x, a, b)` | BatchNorm V2，折叠参数后计算 x*a+b，不做训练统计 |
| `flash_attention(q,k,v,causal,scale,backend,query_start)` | CUDA FP16 D=128，GQA prefill / dense decode，见上文 |

以上 Norm/Softmax 自定义 kernel 尚无 backward；该库以模型推理为目标。
FlashAttention 走独立的推理入口，旧 D=64 实验文件仅作为来源保留。

## 文件与来源

```text
dandan_ops/
  functional.py           # GEMM 后端选择以及其他函数接口
  quantization.py         # INT4 量化格式
  quantized_linear.py     # GEMV / Tensor Core / reference 路由
  _dispatch.py            # 严格环境匹配的离线性能选择
  profiles/              # 精确 M/N/K 的性能表，随 wheel 打包
  integrations/
    myvllm.py             # INT4 状态转换和投影适配
    myvllm_decode.py      # 可选 Graph decode 元数据 slab
  csrc/
    hgemm.cu              # V8 衍生 HGEMM / W4A16 launcher 与注册
    hgemm_tma.cu          # V10 TMA tensor map / launcher 与注册
    gemm_checks.h         # CUDA 元数据/shape/device/stride 检查
    operators.cu          # 原算子和旧 naive W4A16 对照实现
    flash_inference.cu    # 推理 attention 注册、校验和 dense decode / merge
kernels/
  hgemm_pipeline.cuh      # 从 V8 移植并扩展的计算流水线
  hgemm_tma.cuh           # 提取的 V10 kernel
  w4a16_small.cuh         # M<=8 的 packed-weight warp 归约路径
  flash_tensorcore_inference.cuh # 从 V2.1 适配的 D=128 GQA prefill
  ...                    # 原有 Norm、Softmax、FlashAttention 源码
```

V8 来源：`cuda_code/hgemm/hgemm_v8_sm120_mmasync.cu`。
提取时 SHA256：`d056c2d3448da83ca9a6ce104ffd5ac773f876b85831a67b0175a601dfa2c4bd`。
V10 的来源哈希记录在 `kernels/hgemm_tma.cuh` 文件头。

当前全库验证覆盖 CPU API、量化格式、HGEMM/W4A16、FlashAttention 和可选 myvllm 适配。
当前 cu128 全库 312 项 pytest 通过，其中 GEMM 优化回归 23 项，覆盖 small 数值/尾块、
M=1 逐位一致、非默认 stream、CUDA Graph、未对齐、TMA FP16 输出及离线分派的环境隔离。
本轮优化相关 40 项测试通过 memcheck（0 errors），最终 small kernel 的 13 项专项测试
通过 racecheck（0 hazards）；TMA/HGEMM 的竞态检查也已通过。
wheel 位于 `build/wheels/dandan_ops-0.1.0-cp313-cp313-win_amd64.whl`，
新增 Graph/KV 测试 8 项、投影形状测试 5 项。profile 当前包含 98 条记录；wheel 已重新
打包，并验证包内选择表与源码一致，包含 o_proj 的 small 路由。
本轮另加 60 项元数据适配测试（37 CPU、23 CUDA 集成），覆盖请求重排、部分 batch、跨页、
scratch 所有权、共享只读 prefix、chunked prefill、混合采样、非默认 stream、异常恢复和关闭后的原 Graph 回放。
HGEMM/W4A16 的 16 项 CUDA 测试及 FlashAttention 的 57 项专项测试分别在
Compute Sanitizer memcheck/racecheck 下通过，0 errors / 0 hazards。
包含小尺寸、跨 CTA、尾块、不对齐输入、非默认 stream、CUDA Graph、dispatch、错误输入和 autograd fallback。
当前仅有单 GPU，非当前 GPU 的多卡行为未实机验证。
