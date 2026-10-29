# Attention 接口与数值约定

本库提供 dense Attention、packed 变长 prefill 和直接读取分页 KV 的单 token decode。
三条自研 CUDA 路径支持 FP16/BF16、head dimension 64/128/256、GQA、FP32 累加与
online softmax。构建目标要求 SM80+；当前实机验证范围是 Windows / RTX 5080 / SM120 / cu128。
只实现 inference forward；需 autograd 时 `auto` 回退 PyTorch，严格 `cuda` 报错。

## 接口与布局

| API | Q | K/V | 位置语义 |
|---|---|---|---|
| `flash_attention` | `[B,Hq,Q,D]` | `[B,Hkv,K,D]` | `query_start+i` |
| `flash_attention_varlen` | `[total_Q,Hq,D]` | `[total_K,Hkv,D]` | plan 中每条序列的 `query_starts[b]+i` |
| `paged_attention` | `[B,Hq,D]` | `[num_pages,page_size,Hkv,D]` | `context_lens[b]-1`，含当前 token |

要求 `Hq % Hkv == 0`，Q/K/V 同 dtype、同 device，K/V shape 相同。三条 CUDA 路径
均不展开 GQA 的 KV 头，也不创建完整 Q×K 分数矩阵。非连续或非 16B 对齐的输入会复制，
对应 `*_execution_info` 中的 `input_copies` 会列出输入编号，复制成本应计入接入测量。

`backend="auto"` 按支持条件选择；`cuda` 严格调用自研 kernel；`torch` 强制参考路径。
未命中自研 dtype/dimension 的 fallback 不代表对应模型已经完成接入或获得自研加速。
加载的扩展需提供 `attention_contract_version()==1`；旧扩展需重新构建。

公共选项：

- `scale=None` 使用 `D**-0.5`；可显式传有限 FP32 范围内的正值、负值或零。
- `causal=True` 只看 `key_position <= query_position`。Dense 和 packed 要求
  `0 <= query_start <= K-Q`；后缀 prefill 显式传 `K-Q`，普通 prefill 传 0。
- `window_size=W` 包含当前 token 和前 W−1 个 token，只用于 causal 模式；`None` 为全历史。
- `softcap=c` 对缩放后的 logits 做 `c*tanh(logits/c)`，随后应用 mask 和 softmax；
  `None` 关闭，启用时 c 必须为正且在 FP32 normal 范围内。普通 scale 不能代替 softcap。

Q/K/V 必须已完成模型要求的 Q/K Norm 与 RoPE。模型层负责选择位置编码与输出 gate。
本接口尚未实现任意 additive mask、ALiBi、不同 QK/V head dimension 或 MLA。

## Dense

```python
import torch
import dandan_ops as ops

q = torch.randn(1, 16, 17, 256, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 4, 65, 256, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)
with torch.inference_mode():
    out = ops.flash_attention(q, k, v, causal=True, query_start=48,
                              window_size=32, softcap=4.0, backend="cuda")
    print(ops.flash_attention_execution_info(q, k, v))
```

Prefill 使用 `ldmatrix`、FP16/BF16 `mma.sync.m16n8k16`、K/V 双缓冲 `cp.async`。
D64/D128 的 tile 为 64×32；D256 为 64×16，以控制 shared memory 使用量。
Softmax 归约与累加使用 FP32，概率会舍入为输入 dtype 后参与 PV Tensor Core 乘法。
Decode 按 256 个 KV token 分段，warp 归约并在 FP32 合并 softmax 状态。

Dense 的各维必须非空。Tensor 索引限制为 signed int32，`B*Hq <= 65535`。
不同归约顺序及概率舍入会改变低位，不承诺与 SDPA 逐位相等。BF16 的误差容限独立于 FP16；
测试使用显式 FP32 score/softmax oracle，而不只比较自动 SDPA。

## Packed 变长 prefill

```python
plan = ops.prepare_varlen_attention(
    q_lengths=[17, 65], kv_lengths=[33, 129], query_starts=[16, 64], device="cuda")
q = torch.randn(82, 16, 256, device="cuda", dtype=torch.float16)
k = torch.randn(162, 4, 256, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)
with torch.inference_mode():
    out = ops.flash_attention_varlen(q, k, v, plan, causal=True, backend="cuda")
    print(ops.flash_attention_varlen_execution_info(q, k, v, plan))
```

长度和起点在 CPU 完整校验后上传 int32 累积 offsets；forward 不读取 GPU 数值做主机判断。
CUDA grid 在同一次 launch 中处理 packed 请求，不做 padding、不拆为逐请求 Python kernel 调用。
Plan 的序列布局固定；改变长度时重新 prepare。允许零长度请求和全空 batch 的 token 数据，
但 batch metadata 本身至少一行。参考路径逐请求调用 PyTorch，主要用于对拍。

## 分页单 token decode

```python
k_cache = torch.randn(16, 16, 4, 256, device="cuda", dtype=torch.float16)
v_cache = torch.randn_like(k_cache)
plan = ops.prepare_paged_attention(
    block_tables=[[3, 1, -1], [7, 2, 5]], context_lens=[17, 35],
    num_blocks=16, block_size=16, device="cuda", max_context=48)
q = torch.randn(2, 16, 256, device="cuda", dtype=k_cache.dtype)
with torch.inference_mode():
    out = ops.paged_attention(q, k_cache, v_cache, plan, backend="cuda")
    print(ops.paged_attention_execution_info(q, k_cache, v_cache, plan))
```

逻辑 token p 直接定位 `block_tables[b,p//page_size]` 的物理页及 `p%page_size` 槽位。
CUDA 路径不 gather dense KV；参考路径会 gather，不能将其称为分页 CUDA kernel。
Context length 包含已经写入 cache 的当前 token。允许共享只读 prefix 页，允许各请求长度不同，
长度 0 代表 inactive/padding 行，输出全零且不读取 cache。页表未使用的槽位可填 −1。
Active 页 ID 必须合法；算子只读 cache，不负责分配、写入、引用计数或请求归属校验。

Plan 接收 CPU int32/int64 tensor 或 Python 整数序列。所有数值校验在 prepare/update 阶段完成。
输入 shape、容量、索引范围或有效页错误会在提交 metadata 前报错。修改 plan 的私有 device
buffer 会使后续 Python forward 拒绝执行；应只通过 `update` 修改 metadata。

## CUDA Graph 与更新

三条 CUDA 路径均支持非默认 stream 和 CUDA Graph。Prepare、update 必须在 capture 外执行。
Paged plan 的 `max_context` 固定，用于决定 capture 的 KV split 数；实际长度可在该容量内变化。
它默认等于页表宽度×page_size，最大为 65535×256。选择符合实际 bucket 的容量，避免过多空 split。

```python
# q/cache/plan 已创建，先在独立 stream 完成 warmup，然后 capture。
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream), torch.inference_mode():
    for _ in range(3):
        ops.paged_attention(q, k_cache, v_cache, plan, backend="cuda")
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
    captured_out = ops.paged_attention(q, k_cache, v_cache, plan, backend="cuda")

# 在 replay 的同一 stream 上更新；缓冲地址不变。cache 写入也需由引擎按序完成。
plan.update([[3, 1, -1], [-1, -1, -1]], [18, 0])
graph.replay()
```

调用方需保持 q/cache/plan 的存储生命周期和跨 stream 依赖。不要与同一 metadata buffer 的
并发 replay/update 竞争。`update` 当前使用 CPU 校验与常规 H2D 拷贝；尚未优化为引擎已有的
pinned metadata slab。基准排除此项，不能把 kernel 时间当成完整 decode step 延迟。
Paged PyTorch reference 不允许 capture，避免捕获 CPU 常量后忽略后续 plan 更新。

注册的底层 `torch.ops.dandan_ops.flash_varlen/paged_decode` 可供引擎直接使用 GPU metadata，
但引擎必须在提交前完成与 plan 相同的数值和所有权校验，并保持 capture 中的容量约定。
底层边界 guard 用于避免非法读取，不代替公共 API 的报错与校验契约。

## 接入范围与复现

目前已覆盖 dense prefill/decode、packed prefill、paged 单 token decode；尚未实现直接读取分页
KV 的多 query/chunked prefill，也未修改 myvllm 模型或验证 Qwen3.5 完整 logits/生成。
请求调度、KV 写入、Prefix Cache 与混合状态缓存继续由引擎负责。

2026-10-26，在上述 RTX 5080/cu128 环境完成：

- 全库回归 **528 passed、1 skipped**；跳过的是新版引擎不适用的旧 Graph adapter 模块。
- 新增 Attention 测试 **135 passed**，包含两种精度、三种 D、Qwen3.5 的 16Q/4KV 头配置、
  序列与物理页尾块、请求隔离、NaN padding、metadata 异常、梯度回退、非默认 stream 和 Graph 更新。
- Compute Sanitizer memcheck：**135 passed、0 errors**。
- 自研 `dandan_flash` kernel racecheck：**127 passed、8 deselected，0 errors、0 warnings**。
- 独立目录安装 wheel 后，在新进程验证 FP16/BF16 D256、packed prefill、paged decode 及原有算子。

Sanitizer 13.3 使用 `--report-api-errors explicit`；默认 extended 模式会额外记录驱动的
“Selective device code recompilation in progress”提示。设备内存与显式 CUDA API 错误检查保持启用。
数值验证通过不代表已超过 cuDNN/FlashAttention 或获得完整引擎收益。

```powershell
python -m pytest tests/test_flash_inference.py tests/test_attention_extended.py -q
python -m benchmarks.bench_flash_inference --D 256 --dtype bfloat16 --Hkv 4 --shapes 128x128 1x1024
python -m benchmarks.bench_attention_layouts --D 256 --dtype float16
compute-sanitizer --tool memcheck --report-api-errors explicit --error-exitcode 99 python -m pytest tests/test_attention_extended.py -q
compute-sanitizer --tool racecheck --report-api-errors explicit --kernel-name kns=dandan_flash --error-exitcode 99 python -m pytest tests/test_attention_extended.py -q -k "not gradient and not grad and not reference and not cpu"
```

Benchmark 使用重复输入、warm cache、CUDA Graph GPU 时间，包含正常输出和 scratch 工作。
排除 Python dispatch、prepare/update、capture、cache 写入和引擎调度；分页基线的 gather GPU
工作计入时间。输出默认位于忽略的 `build/benchmarks/`，不会增加根目录 JSON。
