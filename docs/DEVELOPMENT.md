# 接口与实现约定

当前安装方法和验证环境见 [README](../README.md)，性能数据与复现命令集中在
[基准说明](BENCHMARKS.md)。本文件保留当前 API、数值与接入约定；逐轮实验日志已归档。
新版 myvllm 已内置元数据缓冲和图内采样，使用其原生 runtime，旧适配器仅用于兼容旧接口。

## 常用算子补充评估

2026-10-29 按当前公开 API、myvllm 的实际调用和维护成本评估。下表属于待实现范围，
不能当作当前支持清单。RoPE、KV 写入、greedy argmax 已有引擎实现可作为移植起点，
本库仍需独立的输入校验、数值约定和回归验证，不能直接把引擎源码变成运行依赖。

| 顺序 | 缺口与用途 | 首轮范围与复杂度 | 验收重点 |
|---|---|---|---|
| P1 | RoPE：补齐 Q/K Norm 到 Attention 的位置编码步骤 | 中；先使用预计算 cos/sin，明确 split-half 布局与 rotary_dim，支持 FP16/BF16 和 GQA | 位置索引、部分旋转时未旋转区不变、非连续输入、stream/Graph；不一次性承诺所有 YaRN/MRoPE 变体 |
| P1 | KV scatter：把当前 K/V 写进分页 cache，与已有 paged decode 连通 | 中；先做明确 slot_mapping 的写入接口，分配和页归属继续由引擎负责 | 非连续物理页、尾块、重复 slot/越界拒绝、只改目标槽位、Graph metadata 更新与 stream 顺序 |
| P1 | greedy argmax：输出 GPU token ids，便于图内 greedy decode | 低到中；先做 temperature=0，复用引擎两阶段归约 | 并列最大值、NaN 约定与 PyTorch 对齐，长词表、dtype、Graph；无每行 .item()/D2H |
| P2 | FP16/BF16 LayerNorm：扩展非 RMSNorm 模型的低精度快速路径 | 中；沿用稳定的方差计算和统一 backend 接口 | 大均值小方差、尾块、FP32 统计、梯度回退；当前 Qwen 主路径优先级低于前三项 |

Embedding 可以继续使用 PyTorch；确认完整调用中的瓶颈后再考虑独立 kernel。
Top-k/top-p 随机采样涉及 RNG、分布正确性与 Graph 状态管理，MoE/grouped GEMM、
DeltaNet/因果卷积涉及额外模型与状态契约，暂不为扩充数量而加入。
零中心/门控 RMSNorm 或 GELU 融合只在目标模型确有调用时增加；避免把模型变体都塞进默认接口。

对照成熟推理库：[FlashInfer RoPE](https://docs.flashinfer.ai/api/rope.html)、
[sampling](https://docs.flashinfer.ai/api/sampling.html)、[norm](https://docs.flashinfer.ai/api/norm.html)。
每个补充项独立验证后再接入模型，分别报告单算子与真实 decode 结果。
沿用现有模块和测试目录，不为每个小功能新建文档或发布脚本。

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

## Inference Attention

提供 dense Attention、packed 变长 prefill，以及直接读取分页 KV 的单 token decode。
三条自研 CUDA 路径支持 FP16/BF16、D64/D128/D256、GQA、causal、滑动窗口与 logit softcap，
使用 FP32 累加；需梯度或不支持的输入由 `auto` 回退 PyTorch，严格 `cuda` 报错。

- `flash_attention(q,k,v,...)`：dense `[B,H,N,D]`，保留原有 `query_start` 调用兼容。
- `flash_attention_varlen(q,k,v,plan,...)`：packed `[total_tokens,H,D]`，prepare CPU 长度与起点。
- `paged_attention(q,k_cache,v_cache,plan,...)`：Q 为 `[B,Hq,D]`，cache 为 `[pages,page_size,Hkv,D]`。
- `window_size=W` 含当前 token 和 W−1 个历史 token，限 causal；`softcap=c` 在 mask/softmax 前
  对缩放 logits 做 `c*tanh(logits/c)`。默认 `None` 关闭相应选项。
- 对应 `*_execution_info` 提供后端、回退原因和输入复制情况。

Plan 在 capture 外校验 CPU metadata，forward 只检查 shape/版本，不扫描 GPU 数据或 D2H 同步。
分页 plan 支持固定地址的 `update()` 和 Graph replay；空请求输出零，缓存只读。
KV 写入、分配、所有权和 stream 顺序由引擎负责。尚未实现分页多 query prefill 或 MLA，
也未替换 myvllm 当前模型路径；dense/packed 的可用性不等于整模型支持。

完整 shape、数值契约、调用示例、Graph 生命周期和接入边界见 [Attention 接口](ATTENTION.md)。

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

## Norm/Softmax 与融合小算子

Small operators use `backend="auto"`, `"cuda"` or `"torch"`. Existing positional
arguments remain compatible. `auto` selects a supported CUDA inference kernel
and falls back to PyTorch for gradient requirements, unsupported dtypes or a
missing extension. `cuda` rejects unsupported inputs; it never silently falls
back. `torch` forces the reference path.

| API | Public input | Custom CUDA types |
|---|---|---|
| `softmax(x, dim=-1)` | PyTorch-compatible floating tensor/dim | FP32 `[M,N]`, last dimension; arbitrary N |
| `rms_norm(x, weight, eps=1e-5)` | `x=[...,D]`, `weight=[D]` | FP16/BF16; FP32 requires D divisible by 4 |
| `layer_norm(x, weight, bias, eps=1e-5)` | `x=[...,D]`, weight/bias `[D]` | FP32 |
| `batch_norm_infer(x, a, b)` | `x=[...,D]`, a/b `[D]`; affine inference transform | FP32 |
| `residual_rms_norm(x, residual, weight, eps=1e-5)` | matching x/residual `[...,D]`, weight `[D]` | FP16/BF16 |
| `silu_and_mul(gate, up)` | matching shape and dtype | FP16/BF16 |

All tensors in an operation must share a device. CUDA requires matching
dtypes; mixed weight/activation dtypes use the PyTorch path where supported.
BF16 CUDA requires SM80+. Only RTX 5080/SM120 has been validated locally.
Normalized D must be positive; empty leading dimensions return empty outputs.
Softmax also permits empty columns. Epsilon must be finite, nonnegative and
representable in FP32. CUDA indexing is bounded to signed 32-bit sizes.

### Numeric contracts

- FP16/BF16 RMSNorm computes squared values, reduction and normalization in
  FP32, casts the normalized activation back to the input dtype, then multiplies
  by weight. This preserves Qwen3's rounding point and avoids FP16 square
  overflow. FP64 reference inputs retain FP64 computation.
- `residual_rms_norm` returns `(normalized, summed)`. Addition rounds to the
  input dtype before the RMSNorm reduction. Both outputs are fresh tensors;
  inputs are not mutated or aliased by the outputs.
- `silu_and_mul` rounds the SiLU result to the activation dtype before multiplying
  by `up`, matching separate eager `F.silu(gate) * up` operations.
- FP32 CUDA LayerNorm computes centered squared differences after the mean,
  avoiding cancellation in `E[x²] - E[x]²`. It may differ slightly from the
  PyTorch FP32 result; adversarial offset tests also compare against FP64.
- CUDA reduction order can change low bits. Cross-backend bitwise identity is
  not promised. Tests check both tensor error and small-model logits/tokens.
- A gradient requirement on any input/weight selects PyTorch in `auto` mode.
  Raw CUDA entry points reject gradient-enabled inputs. Under `no_grad` or
  `inference_mode`, parameters may retain `requires_grad=True`.

### Layout and diagnostics

```python
import dandan_ops as ops

info = ops.elementwise_execution_info("rms_norm", x, weight, backend="auto")
normalized, summed = ops.residual_rms_norm(x, residual, weight, backend="cuda")
activated = ops.silu_and_mul(gate, up, backend="cuda")
```

The report includes `requested_backend`, `implementation`, `reason`, and
`input_copies` (zero-based positional tensor indices). The latter describes
explicit input copies, not outputs or PyTorch-internal temporaries. Strided
inputs are copied to contiguous storage; an already-contiguous but unaligned
FP32 RMSNorm view is cloned to obtain 16-byte alignment. Low-precision kernels
accept offset views and use scalar access when vector alignment is unavailable.
Public Norm wrappers flatten leading dimensions and restore the original shape.

`gemm_execution_info`, `flash_attention_execution_info` and
`QuantizedLinear.execution_info` also expose `input_copies`. Quantized reports
retain the old `fallback_reason` field and add `reason` for consistency.
All diagnostics inspect metadata only and do not scan tensor values or time
kernels. CUDA Graph capture is covered; this does not imply `torch.compile`
or FakeTensor support.

## 可选 decode 元数据缓冲适配

`dandan_ops.integrations.myvllm_decode.install_buffered_decode(runtime)` 只可安装到 `ae967dd` 旧版
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
