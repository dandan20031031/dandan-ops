"""Opt-in metadata buffers for myvllm's existing synchronous Graph runtime.

Install before capturing graphs. The scheduler, prefill and sampling stay in
the engine. Each bucket gets a pinned CPU slab and a fixed GPU slab, laid out
as contiguous tokens[B], positions[B], tables[B,W]. Replay uploads it once.
This executor is single-stream and synchronous: sampled tokens return to CPU
before the scheduler can release pages or the next call can reuse a buffer.
"""
import operator

import torch


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_padding(padding, batch_size, max_blocks, num_blocks):
    if len(padding) < max(batch_size,max_blocks):
        raise ValueError("scratch pages must cover batch_size and max_blocks")
    private = tuple(_integer(p,"scratch page") for p in padding)
    if len(set(private[:batch_size])) != batch_size or any(p < 0 or p >= num_blocks for p in private):
        raise ValueError("scratch pages require valid distinct private pages")
    return private


def _validate_metadata(requests, *, batch_size, max_blocks, padding,
                       block_size, num_blocks, vocab_size=None, max_position=None):
    active = len(requests)
    if not 0 < active <= batch_size:
        raise ValueError("active batch exceeds the metadata bucket")
    private = _validate_padding(padding,batch_size,max_blocks,num_blocks)
    scratch = set(private)
    encoded, write_pages, page_uses = [], [], {}
    # Validate the entire batch before any upload or live-KV write.
    for request in requests:
        if request.is_finished or not request.output_token_ids:
            raise ValueError("decode requires an active request with a previous token")
        token = _integer(request.output_token_ids[-1],"token")
        position = _integer(request.total_len-1,"position")
        pages = tuple(_integer(p,"physical page") for p in request.block_ids)
        if token < 0 or (vocab_size is not None and token >= vocab_size):
            raise ValueError("decode token is outside the vocabulary")
        if position < 0 or (max_position is not None and position >= max_position):
            raise ValueError("decode position exceeds the model context")
        if not pages or len(pages) > max_blocks or position // block_size >= len(pages):
            raise ValueError("block table does not cover the decode position")
        if any(p < 0 or p >= num_blocks for p in pages):
            raise ValueError("physical page is outside the KV cache")
        if scratch.intersection(pages):
            raise ValueError("live and scratch pages must be disjoint")
        write_pages.append(pages[position//block_size])
        for page in pages:
            page_uses[page] = page_uses.get(page,0) + 1
        encoded.append((token,position,pages))
    if any(page_uses[page] != 1 for page in write_pages):
        raise ValueError("each decode write page must be private to its logical page")
    return private,encoded


def _fill_validated(array,encoded,private,batch_size,max_blocks):
    tokens = array[:batch_size]
    positions = array[batch_size:2*batch_size]
    tables = array[2*batch_size:].reshape(batch_size,max_blocks)
    tokens.fill(0)
    positions.fill(0)
    tables[:] = private[:max_blocks]
    tables[:,0] = private[:batch_size]
    for row,(token,position,pages) in enumerate(encoded):
        tokens[row] = token
        positions[row] = position
        tables[row].fill(-1)
        tables[row,:len(pages)] = pages


def fill_decode_metadata(array, requests, *, batch_size, max_blocks, padding,
                         block_size, num_blocks, vocab_size=None, max_position=None):
    """Fill every row after validating the whole batch; inspect CPU metadata.

    array is the NumPy view of the pinned int64 slab. No slot/request identity
    cache is used: the scheduler may reorder requests and recycle pages.
    Shared read-only prefix pages are allowed; current write pages are private.
    """
    private,encoded = _validate_metadata(requests,batch_size=batch_size,
        max_blocks=max_blocks,padding=padding,block_size=block_size,
        num_blocks=num_blocks,vocab_size=vocab_size,max_position=max_position)
    _fill_validated(array,encoded,private,batch_size,max_blocks)


class _Workspace:
    def __init__(self,bucket):
        b,w = bucket.shape.batch_size,bucket.shape.max_blocks
        self.bucket = bucket
        self.host = torch.empty(b*(2+w),dtype=torch.long,device="cpu",pin_memory=True)
        self.array = self.host.numpy()
        self.device = torch.empty_like(self.host,device=bucket.device)
        self.copy_done = torch.cuda.Event()
        self.uploaded = False
        self.uploads = 0
        bucket.input_ids = self.device[:b]
        bucket.positions = self.device[b:2*b]
        bucket.block_tables = self.device[2*b:].view(b,w)


class BufferedDecode:
    """Installed by install_buffered_decode; call close() to restore decode.

    Existing captured buckets cannot be rebound safely and are rejected.
    Install on the stream that will run runtime.step(). Do not change weights,
    Graph bucket inputs, scratch configuration or streams while installed.
    """
    def __init__(self,runtime,*,scratch_owner="__graph_padding__"):
        if hasattr(runtime.graph_manager,"capture_sampling"):
            raise ValueError("this adapter targets the legacy Graph API; newer myvllm already "
                             "buffers metadata and captures sampling, so use its native runtime")
        if runtime.graph_manager is None or runtime.cache.device.type != "cuda":
            raise ValueError("buffered decode requires a CUDA Graph runtime")
        if not callable(getattr(runtime,"_sample",None)):
            raise ValueError("buffered decode requires runtime._sample")
        if any(bucket.captured for bucket in runtime.graph_manager.buckets.values()):
            raise ValueError("install buffered decode before capturing Graph buckets")
        padding = runtime.graph_padding_block_ids
        if padding is None:
            raise ValueError("reserve private graph_padding_block_ids first")
        if isinstance(padding,torch.Tensor):
            padding = padding.cpu().tolist()
        self.padding = _validate_padding(padding,
            max(runtime.graph_manager.batch_sizes),max(runtime.graph_manager.block_buckets),
            runtime.cache.shape.num_blocks)
        self.padding_source = runtime.graph_padding_block_ids
        self.padding_version = None
        if isinstance(self.padding_source,torch.Tensor):
            try:
                self.padding_version = self.padding_source._version
            except RuntimeError as exc:
                raise ValueError("use CPU page ids for scratch tensors without a version counter") from exc
        self.runtime = runtime
        self.scratch_owner = scratch_owner
        self._check_scratch_owner()
        self.stream = torch.cuda.current_stream(runtime.cache.device)
        self.original = runtime.decode_step
        self.workspaces = {}
        self.closed = False
        self.busy = False
        self.steps = 0

    def _check_stream(self):
        if torch.cuda.current_stream(self.runtime.cache.device).cuda_stream != self.stream.cuda_stream:
            raise ValueError("buffered decode must use its installation stream")

    def _check_padding(self):
        current = self.runtime.graph_padding_block_ids
        if current is not self.padding_source:
            raise ValueError("scratch configuration changed; create a new runtime")
        changed = (current._version != self.padding_version if isinstance(current,torch.Tensor)
                   else tuple(current) != self.padding)
        if changed:
            raise ValueError("scratch configuration changed; create a new runtime")

    def _check_scratch_owner(self):
        manager = self.runtime.scheduler.block_manager
        try:
            reserved = set(manager.table(self.scratch_owner).block_ids)
        except KeyError as exc:
            raise ValueError("scratch pages must have a retained allocator owner") from exc
        active = list(self.runtime.scheduler.waiting) + list(self.runtime.scheduler.running.values())
        if (any(request.request_id == self.scratch_owner for request in active)
                or not set(self.padding).issubset(reserved)
                or any(manager.refcount(page) != 1 for page in self.padding)):
            raise ValueError("scratch owner must retain private pages outside live requests/prefix cache")

    def _synchronize_after_error(self,error):
        try:
            self.stream.synchronize()
        except Exception as sync_error:
            # Keep the original failure useful even when CUDA itself is faulty.
            if hasattr(error,"add_note"):
                error.add_note(f"CUDA cleanup synchronization also failed: {sync_error}")

    @torch.inference_mode()
    def prepare(self,batch_size,max_blocks,*,warmup=3):
        """Allocate/bind/capture a bucket outside a latency measurement."""
        if self.closed:
            raise RuntimeError("buffered decode is closed")
        self._check_stream()
        self._check_padding()
        manager = self.runtime.graph_manager
        shape = manager.shape_for(batch_size,max_blocks)
        if shape in self.workspaces:
            return self.workspaces[shape].bucket
        self._check_scratch_owner()
        existing = manager.buckets.get(shape)
        bucket = manager.get(batch_size,max_blocks,padding_block_ids=self.padding)
        if bucket.captured:
            raise ValueError("cannot replace storage of an externally captured bucket")
        old = bucket.input_ids,bucket.positions,bucket.block_tables,bucket.logits
        try:
            workspace = _Workspace(bucket)
            bucket.capture(warmup=warmup)
        except Exception as error:
            try:
                self._synchronize_after_error(error)
            finally:
                bucket.input_ids,bucket.positions,bucket.block_tables,bucket.logits = old
                bucket.graph = None
                if existing is None and manager.buckets.get(shape) is bucket:
                    del manager.buckets[shape]
            raise
        self.workspaces[shape] = workspace
        return bucket

    @torch.inference_mode()
    def decode_step(self,requests):
        if self.closed:
            raise RuntimeError("buffered decode is closed")
        if not requests:
            return {}
        if self.busy:
            raise RuntimeError("buffered decode does not support concurrent calls")
        self._check_stream()
        self._check_padding()
        self.busy = True
        queued = False
        try:
            width = max(len(request.block_ids) for request in requests)
            shape = self.runtime.graph_manager.shape_for(len(requests),width)
            cfg = getattr(self.runtime.model,"config",None)
            # Validate before lazy capture can write even reserved scratch KV.
            private,encoded = _validate_metadata(requests,
                batch_size=shape.batch_size,max_blocks=shape.max_blocks,
                padding=self.padding,block_size=self.runtime.cache.shape.block_size,
                num_blocks=self.runtime.cache.shape.num_blocks,
                vocab_size=getattr(cfg,"vocab_size",None),
                max_position=getattr(cfg,"max_position_embeddings",None))
            bucket = self.prepare(len(requests),width)
            workspace = self.workspaces[bucket.shape]
            if workspace.uploaded:
                # CUDA stream ordering alone cannot protect CPU writes to the
                # pinned source buffer while the previous DMA is reading it.
                workspace.copy_done.synchronize()
            _fill_validated(workspace.array,encoded,private,
                            bucket.shape.batch_size,bucket.shape.max_blocks)
            workspace.device.copy_(workspace.host,non_blocking=True)
            queued = True
            workspace.copy_done.record(self.stream)
            workspace.uploaded = True
            workspace.uploads += 1
            # Inputs already reference the slab. Calling bucket.replay here
            # would reintroduce three input copies and padding kernels.
            bucket.graph.replay()
            logits = bucket.logits[:len(requests)]
            # Keep current engine sampling semantics, including mixed params,
            # and its existing single batch D2H before releasing any KV pages.
            chosen = self.runtime._sample(logits,requests).cpu().tolist()
            if len(chosen) != len(requests):
                raise ValueError("sampling must return one token per active request")
            result = {}
            for request,token in zip(requests,chosen):
                token = int(token)
                self.runtime.scheduler.on_token(request,token,finished=token in self.runtime.eos_token_ids)
                result[request.request_id] = token
            self.steps += 1
            return result
        except Exception as error:
            if queued:
                # The engine aborts/recycles pages when this exception escapes.
                # Finish already-submitted GPU work before that can happen.
                self._synchronize_after_error(error)
            raise
        finally:
            self.busy = False

    def execution_info(self):
        return {"steps":self.steps,"metadata_uploads":sum(w.uploads for w in self.workspaces.values()),
                "sampling":"runtime._sample; one batch CPU transfer", "stream":int(self.stream.cuda_stream),
                "scratch_owner":self.scratch_owner,
                "buckets":[{"batch":s.batch_size,"max_blocks":s.max_blocks,
                    "host_ptr":w.host.data_ptr(),"device_ptr":w.device.data_ptr(),
                    "pinned":w.host.is_pinned(),"bytes":w.host.numel()*w.host.element_size()}
                    for s,w in self.workspaces.items()]}

    def close(self):
        if self.closed:
            return
        if self.busy:
            raise RuntimeError("cannot close buffered decode during a call")
        try:
            self.stream.synchronize()
        finally:
            if self.runtime.decode_step == self.decode_step:
                self.runtime.decode_step = self.original
            if getattr(self.runtime,"_dandan_buffered_decode",None) is self:
                del self.runtime._dandan_buffered_decode
            # Bucket views keep GPU storage alive for the restored replay path.
            self.workspaces.clear()
            self.closed = True

    def __enter__(self): return self
    def __exit__(self,exc_type,exc,traceback):
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except Exception as close_error:
                if hasattr(exc,"add_note"):
                    exc.add_note(f"Buffered decode close also failed: {close_error}")


def install_buffered_decode(runtime, *, scratch_owner="__graph_padding__"):
    """Attach before capture; scratch_owner must retain the private KV pages."""
    if getattr(runtime,"_dandan_buffered_decode",None) is not None:
        raise ValueError("buffered decode is already installed")
    executor = BufferedDecode(runtime,scratch_owner=scratch_owner)
    runtime.decode_step = executor.decode_step
    runtime._dandan_buffered_decode = executor
    return executor
