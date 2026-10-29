"""Inference attention and CPU-validated, persistent batch metadata.

Plans are prepared outside capture. Forward checks metadata/versions only:
no device value scans, D2H copies or host synchronization. A paged plan's
update() validates CPU data before copying into fixed device buffers.
"""
import itertools
import math
import operator

import torch
import torch.nn.functional as F

from ._extension import load_extension

_INT_MAX = 2**31 - 1


def _integer(value, name, minimum=0, maximum=_INT_MAX):
    try:
        if isinstance(value, bool):
            raise TypeError
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} outside [{minimum}, {maximum}]")
    return value


def _options(scale, window_size, softcap, causal, d):
    scale = d**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or abs(scale) > torch.finfo(torch.float32).max:
        raise ValueError("scale must be finite and representable as FP32")
    window = -1 if window_size is None else _integer(window_size, "window_size", 1)
    if window > 0 and not causal:
        raise ValueError("window_size requires causal attention")
    cap = 0.0 if softcap is None else float(softcap)
    if softcap is not None and (not math.isfinite(cap) or not torch.finfo(torch.float32).tiny <= cap <= torch.finfo(torch.float32).max):
        raise ValueError("softcap must be positive, finite and representable as FP32")
    return scale, window, cap


def _tensors(q, k, v, qrank, krank, kv_head_axis):
    if any(not isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise ValueError("q, k and v must be tensors")
    if q.ndim != qrank or k.ndim != krank or v.shape != k.shape:
        raise ValueError("invalid attention ranks or matching K/V shapes")
    if qrank == 4 and any(size == 0 for t in (q, k, v) for size in t.shape):
        raise ValueError("attention dimensions must be non-empty")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ValueError("attention tensors must share dtype and device")
    if not q.is_floating_point():
        raise ValueError("attention requires floating-point tensors")
    if q.shape[1] <= 0 or k.shape[kv_head_axis] <= 0 or q.shape[-1] <= 0 or k.shape[-1] != q.shape[-1] or q.shape[1] % k.shape[kv_head_axis]:
        raise ValueError("invalid GQA head dimensions")


def _route(q, k, v, backend, batches):
    if backend not in ("auto", "cuda", "torch"):
        raise ValueError("attention backend must be auto, cuda, or torch")
    reason = None
    if backend == "torch":
        reason = "explicit torch requested"
    elif (not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16)
          or q.shape[-1] not in (64, 128, 256)
          or (q.is_cuda and torch.cuda.get_device_capability(q.device)[0] < 8)):
        reason = "CUDA attention requires SM80+, FP16/BF16 and head_dim=64,128,256"
    elif torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        reason = "CUDA attention requires forward inference"
    elif max(q.numel(), k.numel()) > _INT_MAX or batches * q.shape[1] > 65535:
        reason = "CUDA attention index/grid range exceeded"
    elif not load_extension(required=backend == "cuda"):
        reason = "CUDA extension unavailable"
    elif not hasattr(torch.ops.dandan_ops, "attention_contract_version") or torch.ops.dandan_ops.attention_contract_version() != 1:
        reason = "stale CUDA extension; rebuild for attention contract version 1"
    if reason is not None and backend == "cuda":
        raise ValueError(reason)
    route = "cuda" if reason is None else "torch"
    copies = [i for i, t in enumerate((q, k, v)) if not t.is_contiguous() or t.data_ptr() % 16] if route == "cuda" else []
    return {"requested_backend": backend, "implementation": route,
            "reason": reason or "supported inference input", "input_copies": copies}


def _aligned(x):
    x = x.contiguous()
    return x.clone() if x.data_ptr() % 16 else x


def _reference(q, k, v, causal, start, scale, window, cap):
    mask = None
    # Preserve SDPA's native causal dispatch for ordinary prefill and its
    # unmasked dispatch for a suffix decode query that sees the entire cache.
    # Explicit masks are needed only for windows, suffix blocks or softcap.
    full_decode = q.shape[-2] == 1 and start == k.shape[-2] - 1
    native_causal = bool(causal) and start == 0 and window < 0
    if causal and (cap or window > 0 or (start != 0 and not full_decode)):
        pos = torch.arange(q.shape[-2], device=q.device)[:, None] + start
        keys = torch.arange(k.shape[-2], device=q.device)[None, :]
        mask = keys <= pos
        if window > 0:
            mask = mask & (keys >= pos - window + 1)
    if not cap:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            is_causal=native_causal and mask is None, scale=scale, enable_gqa=True)
    # SDPA does not expose logit softcapping. Use an explicit reference only
    # for this option; custom kernels still stream and never allocate Q*K.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    repeat = q.shape[1] // k.shape[1]
    logits = q.to(dtype) @ k.to(dtype).repeat_interleave(repeat, 1).transpose(-1, -2)
    logits = torch.tanh(logits * scale / cap) * cap
    if mask is not None:
        logits = logits.masked_fill(~mask, -float("inf"))
    return (logits.softmax(-1) @ v.to(dtype).repeat_interleave(repeat, 1)).to(q.dtype)


def flash_attention_execution_info(q, k, v, causal=False, scale=None, backend="auto", query_start=0,
                                   *, window_size=None, softcap=None):
    """Dense Q=[B,Hq,Q,D], K/V=[B,Hkv,K,D]; absolute query positions.

    CUDA: FP16/BF16 D64/128/256, SM80+, inference only. A causal window W
    includes the current token and W-1 predecessors. Softcap is applied to
    scaled logits before masking/softmax. Metadata inspection only.
    """
    _tensors(q, k, v, 4, 4, 1)
    if any(size == 0 for t in (q, k, v) for size in t.shape):
        raise ValueError("attention dimensions must be non-empty")
    if q.shape[0] != k.shape[0]:
        raise ValueError("attention batch dimensions must match")
    start = _integer(query_start, "query_start")
    if start + q.shape[2] > k.shape[2]:
        raise ValueError("query_start must map Q rows into K rows")
    scale, window, cap = _options(scale, window_size, softcap, causal, q.shape[-1])
    info = _route(q, k, v, backend, q.shape[0])
    return dict(info, query_start=start, scale=scale, window_size=window, softcap=cap)


def flash_attention(q, k, v, causal=False, scale=None, backend="auto", query_start=0,
                    *, window_size=None, softcap=None):
    info = flash_attention_execution_info(q, k, v, causal, scale, backend, query_start,
                                          window_size=window_size, softcap=softcap)
    if info["implementation"] == "cuda":
        return torch.ops.dandan_ops.flash_inference(_aligned(q), _aligned(k), _aligned(v),
            info["query_start"], bool(causal), info["scale"], info["window_size"], info["softcap"])
    return _reference(q, k, v, causal, info["query_start"], info["scale"], info["window_size"], info["softcap"])


def _device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _outside_capture(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("prepare/update attention metadata outside CUDA Graph capture")


def _cpu_sequence(values, name):
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} requires a CPU integer sequence")
        values = values.tolist()
    try:
        return tuple(_integer(x, name) for x in values)
    except TypeError as exc:
        raise ValueError(f"{name} requires a CPU integer sequence") from exc


def _metadata(values, device):
    # Version counters remain available even when callers prepare inside
    # torch.inference_mode(). Forward never inspects device values.
    with torch.inference_mode(False):
        return torch.tensor(values, dtype=torch.int32, device=device)


class VarlenAttentionPlan:
    """Immutable sequence layout; construct with prepare_varlen_attention()."""
    def __init__(self, q_lengths, kv_lengths=None, query_starts=None, *, device="cpu"):
        self._device = _device(device)
        _outside_capture(self.device)
        self._q_lengths = _cpu_sequence(q_lengths, "q_lengths")
        self._kv_lengths = self._q_lengths if kv_lengths is None else _cpu_sequence(kv_lengths, "kv_lengths")
        self._query_starts = (0,) * len(self._q_lengths) if query_starts is None else _cpu_sequence(query_starts, "query_starts")
        if not self._q_lengths or len(self._q_lengths) != len(self._kv_lengths) or len(self._q_lengths) != len(self._query_starts):
            raise ValueError("varlen metadata requires equally sized nonempty sequences")
        if any(s + q > k for q, k, s in zip(self._q_lengths, self._kv_lengths, self._query_starts)):
            raise ValueError("query_starts must map each query into its KV sequence")
        self._q_offsets = tuple(itertools.accumulate(self._q_lengths, initial=0))
        self._k_offsets = tuple(itertools.accumulate(self._kv_lengths, initial=0))
        if max(self._q_offsets[-1], self._k_offsets[-1]) > _INT_MAX:
            raise ValueError("packed token count exceeds int32 range")
        self._buffers = tuple(_metadata(x, self.device) for x in (self._q_offsets, self._k_offsets, self._query_starts))
        self._versions = tuple(t._version for t in self._buffers)

    @property
    def device(self):
        return self._device

    @property
    def q_lengths(self):
        return self._q_lengths

    @property
    def kv_lengths(self):
        return self._kv_lengths

    def _check(self, device):
        if device != self.device:
            raise ValueError("attention plan and query must share device")
        if tuple(t._version for t in self._buffers) != self._versions:
            raise ValueError("attention plan buffers were modified; prepare a new plan")


def prepare_varlen_attention(q_lengths, kv_lengths=None, query_starts=None, *, device="cpu"):
    """Validate CPU lengths and upload persistent int32 packed offsets.

    Zero-length rows are allowed. Defaults describe ordinary prefill;
    suffix prefill supplies query_starts=kv_lengths-q_lengths explicitly.
    """
    return VarlenAttentionPlan(q_lengths, kv_lengths, query_starts, device=device)


def flash_attention_varlen_execution_info(q, k, v, plan, *, causal=True, scale=None, backend="auto",
                                          window_size=None, softcap=None):
    _tensors(q, k, v, 3, 3, 1)
    if not isinstance(plan, VarlenAttentionPlan):
        raise TypeError("plan must come from prepare_varlen_attention")
    plan._check(q.device)
    if q.shape[0] != plan._q_offsets[-1] or k.shape[0] != plan._k_offsets[-1]:
        raise ValueError("packed token counts do not match plan")
    scale, window, cap = _options(scale, window_size, softcap, causal, q.shape[-1])
    return dict(_route(q, k, v, backend, len(plan.q_lengths)), scale=scale, window_size=window, softcap=cap)


def flash_attention_varlen(q, k, v, plan, *, causal=True, scale=None, backend="auto",
                           window_size=None, softcap=None):
    """Packed Q=[total_Q,Hq,D], K/V=[total_K,Hkv,D]; no padding/unpacking."""
    info = flash_attention_varlen_execution_info(q,k,v,plan,causal=causal,scale=scale,
        backend=backend,window_size=window_size,softcap=softcap)
    scale, window, cap = info["scale"], info["window_size"], info["softcap"]
    if info["implementation"] == "cuda":
        return torch.ops.dandan_ops.flash_varlen(_aligned(q), _aligned(k), _aligned(v),
            *plan._buffers, max(plan.q_lengths), bool(causal), scale, window, cap)
    if q.shape[0] == 0:
        return q.clone()
    outputs = []
    for b, length in enumerate(plan.q_lengths):
        qb = q[plan._q_offsets[b]:plan._q_offsets[b+1]].transpose(0, 1).unsqueeze(0)
        kb = k[plan._k_offsets[b]:plan._k_offsets[b+1]].transpose(0, 1).unsqueeze(0)
        vb = v[plan._k_offsets[b]:plan._k_offsets[b+1]].transpose(0, 1).unsqueeze(0)
        if length:
            outputs.append(_reference(qb, kb, vb, causal, plan._query_starts[b], scale, window, cap).squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


class PagedAttentionPlan:
    """CPU-validated block tables with fixed addresses for decode Graphs.

    Metadata buffers are private. Only update() may modify them. The engine
    owns cache pages, request scheduling and stream ordering. Context length
    zero denotes an inactive row and returns zeros without cache reads.
    """
    def __init__(self, block_tables, context_lens, *, num_blocks, block_size, device="cpu", max_context=None):
        self._device = _device(device)
        _outside_capture(self.device)
        self._num_blocks = _integer(num_blocks, "num_blocks", 1)
        self._block_size = _integer(block_size, "block_size", 1)
        tables = self._tables(block_tables)
        if not tables or not tables[0] or any(len(row) != len(tables[0]) for row in tables):
            raise ValueError("block_tables must be a nonempty rectangular CPU integer matrix")
        self._batch, self._width = len(tables), len(tables[0])
        capacity = self._width * self._block_size
        self._max_context = _integer(capacity if max_context is None else max_context, "max_context", 0, min(capacity, 65535*256))
        lengths = self._validate(tables, context_lens)
        self._buffers = (_metadata(tables, self.device), _metadata(lengths, self.device))
        self._versions = tuple(t._version for t in self._buffers)
        self._host_tables, self._lengths = tables, lengths
        self._valid = True

    @staticmethod
    def _tables(values):
        if isinstance(values, torch.Tensor):
            if values.device.type != "cpu" or values.ndim != 2 or values.dtype not in (torch.int32, torch.int64):
                raise ValueError("block_tables requires a CPU integer matrix")
            values = values.tolist()
        try:
            return tuple(tuple(_integer(x, "block_tables", -1) for x in row) for row in values)
        except TypeError as exc:
            raise ValueError("block_tables requires a CPU integer matrix") from exc

    def _validate(self, tables, context_lens):
        lengths = _cpu_sequence(context_lens, "context_lens")
        if len(tables) != self._batch or len(lengths) != self._batch or any(len(row) != self._width for row in tables):
            raise ValueError("paged metadata shapes cannot change; create a new plan")
        for row, length in zip(tables, lengths):
            if length > self._max_context:
                raise ValueError("context_lens exceeds max_context")
            used = (length + self._block_size - 1) // self._block_size
            if any(page < 0 or page >= self._num_blocks for page in row[:used]):
                raise ValueError("active block_tables entries must index physical cache pages")
            if any(page >= self._num_blocks for page in row[used:]):
                raise ValueError("unused block_tables entries must be -1 or valid page IDs")
        return lengths

    @property
    def device(self):
        return self._device

    @property
    def context_lens(self):
        return self._lengths

    def _check(self, device):
        if device != self.device:
            raise ValueError("attention plan and query must share device")
        if not self._valid or tuple(t._version for t in self._buffers) != self._versions:
            raise ValueError("paged plan buffers were modified outside update()")

    def update(self, block_tables, context_lens):
        """Validate CPU metadata and copy into existing buffers outside capture.

        Call on the same CUDA stream as attention/replay. Fixed max_context
        determines split count at capture, so shorter/longer rows can replay.
        """
        self._check(self.device)
        _outside_capture(self.device)
        tables = self._tables(block_tables)
        lengths = self._validate(tables, context_lens)
        self._valid = False
        with torch.no_grad():
            for dst, values in zip(self._buffers, (tables, lengths)):
                dst.copy_(torch.tensor(values, dtype=torch.int32))
        self._versions = tuple(t._version for t in self._buffers)
        self._host_tables, self._lengths = tables, lengths
        self._valid = True


def prepare_paged_attention(block_tables, context_lens, *, num_blocks, block_size,
                            device="cpu", max_context=None):
    return PagedAttentionPlan(block_tables, context_lens, num_blocks=num_blocks,
                               block_size=block_size, device=device, max_context=max_context)


def paged_attention_execution_info(q, k_cache, v_cache, plan, *, scale=None, window_size=None, softcap=None, backend="auto"):
    _tensors(q, k_cache, v_cache, 3, 4, 2)
    if not isinstance(plan, PagedAttentionPlan):
        raise TypeError("plan must come from prepare_paged_attention")
    plan._check(q.device)
    if q.shape[0] != plan._batch or k_cache.shape[:2] != (plan._num_blocks, plan._block_size):
        raise ValueError("query/cache shape does not match paged plan")
    scale, window, cap = _options(scale, window_size, softcap, True, q.shape[-1])
    return dict(_route(q,k_cache,v_cache,backend,q.shape[0]), scale=scale, window_size=window, softcap=cap)


def paged_attention(q, k_cache, v_cache, plan, *, scale=None, window_size=None, softcap=None, backend="auto"):
    """Read-only single-token decode. Q=[B,Hq,D], cache=[pages,page_size,Hkv,D].

    CUDA reads physical pages directly, without gathering dense KV. Shared
    read-only pages are allowed; allocation, ownership and writes belong to
    the engine. This API does not implement paged multi-query prefill.
    """
    info = paged_attention_execution_info(q,k_cache,v_cache,plan,scale=scale,window_size=window_size,
                                         softcap=softcap,backend=backend)
    scale, window, cap = info["scale"], info["window_size"], info["softcap"]
    if info["implementation"] == "cuda":
        return torch.ops.dandan_ops.paged_decode(_aligned(q), _aligned(k_cache), _aligned(v_cache),
            *plan._buffers, plan._max_context, scale, window, cap)
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise ValueError("paged reference capture cannot replay updated CPU metadata; use backend='cuda'")
    outputs = []
    for b, length in enumerate(plan.context_lens):
        if not length:
            # Keep a zero derivative for autograd fallback, including a batch
            # consisting entirely of inactive rows; NaN padding stays hidden.
            qb = q[b:b+1]
            outputs.append(qb.masked_fill(torch.ones_like(qb,dtype=torch.bool),0))
            continue
        used = (length + plan._block_size - 1) // plan._block_size
        pages = list(plan._host_tables[b][:used])
        kb = k_cache[pages].reshape(-1, k_cache.shape[2], q.shape[-1])[:length].transpose(0, 1).unsqueeze(0)
        vb = v_cache[pages].reshape(-1, v_cache.shape[2], q.shape[-1])[:length].transpose(0, 1).unsqueeze(0)
        outputs.append(_reference(q[b:b+1].unsqueeze(2), kb, vb, True, length-1, scale, window, cap).squeeze(2))
    return torch.cat(outputs)
