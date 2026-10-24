"""Optional real-model tests for the process-local buffered decode adapter.

The engine is imported read-only and its existing CUDA extension is loaded
without invoking the build-on-demand path.  No model downloads are needed.
"""

import os
from pathlib import Path
import sys

import pytest
import torch

import dandan_ops as ops


_LIBRARY_ROOT = Path(ops.__file__).resolve().parents[1]
_ENGINE_ROOT = Path(os.environ.get("MYVLLM_ROOT", _LIBRARY_ROOT.parent / "myvllm"))
if not (_ENGINE_ROOT / "mini_engine").is_dir():
    pytest.skip("optional myvllm checkout not found", allow_module_level=True)
sys.path.insert(0, str(_ENGINE_ROOT))
sys.path.insert(0, str(_LIBRARY_ROOT))

from dandan_ops.integrations.myvllm import install_int4_projections
from dandan_ops.integrations.myvllm_decode import install_buffered_decode
from mini_engine.config import QuantConfig
from mini_engine.engine.cuda_graph import DecodeGraphBucket
from mini_engine.engine.request import GenerationRequest, RequestState
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config

if hasattr(DecodeGraphBucket, "replay_metadata"):
    pytest.skip("legacy adapter integration: current myvllm has native metadata buffering and sampling; "
                "use engine ae967dd for this suite", allow_module_level=True)


@pytest.fixture
def graph_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("dandan inference kernels require SM80 or newer")
    from dandan_ops._extension import load_extension
    import mini_engine.kernels.cuda_extension as engine_extension
    from bench_qwen3_graph import load_prebuilt_extension

    monkeypatch.setattr(engine_extension, "load_cuda_extension",
                        engine_extension.load_cuda_extension)
    try:
        load_prebuilt_extension(_ENGINE_ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"optional prebuilt myvllm extension not found: {exc}")
    load_extension(required=True)


def _model(quantized=False):
    torch.manual_seed(109)
    config = Qwen3Config(71, 128, 256, 2, 4, 2, 128,
                         max_position_embeddings=64)
    model = NativeQwen3(config, prefill_backend="math", decode_backend="cuda",
                        data_backend="cuda", elementwise_backend="cuda")
    model.cuda().half().eval()
    if quantized:
        install_int4_projections(
            model, QuantConfig("int4", activation_dtype="fp16", group_size=128)
        )
    return model


def _runtime(model, batch_sizes=(1, 2, 4, 8), block_buckets=(1, 2, 3), *,
             enable_prefix_cache=False, prefill_chunk_size=None):
    class Tokenizer:
        eos_token_id = None

    runner = NativeQwen3Runner(model, Tokenizer(), num_blocks=128, block_size=4)
    runner.cache.clear()
    return runner.create_batch_runtime(
        use_cuda_graph=True, graph_batch_sizes=batch_sizes,
        graph_block_buckets=block_buckets, enable_prefix_cache=enable_prefix_cache,
        prefill_chunk_size=prefill_chunk_size,
    )


def _add(runtime, request_id, length=3, outputs=8, **sampling):
    prompt = [(index + 1 + 3 * len(request_id)) % 71 for index in range(length)]
    request = GenerationRequest(request_id, prompt, max_new_tokens=outputs, **sampling)
    runtime.scheduler.add_request(request)
    return request


def _observe_sampling(runtime):
    original = runtime._sample
    calls = []

    def sample(logits, requests):
        calls.append((tuple(request.request_id for request in requests), logits.clone()))
        return original(logits, requests)

    runtime._sample = sample
    return calls


def _ptrs(executor):
    return {(row["batch"], row["max_blocks"]): (row["host_ptr"], row["device_ptr"])
            for row in executor.execution_info()["buckets"]}


def _assert_requests_and_kv(reference, buffered, expected, actual):
    for left, right in zip(expected, actual):
        assert left.output_token_ids == right.output_token_ids
        assert left.generated_tokens == right.generated_tokens
        assert left.state is right.state
        for name in ("key", "value"):
            left_pages = getattr(reference.cache, name)[:, left.block_ids]
            right_pages = getattr(buffered.cache, name)[:, right.block_ids]
            # Full pages include unwritten positions and future pages, so a
            # padding write into a currently inactive request also fails.
            torch.testing.assert_close(right_pages, left_pages, atol=0.005, rtol=0.01)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("quantized", [False, True], ids=["fp16", "int4"])
@torch.inference_mode()
def test_buffered_decode_matches_original_with_reordering(graph_cuda, batch_size, quantized):
    model = _model(quantized)
    # A single batch bucket ensures partial batches exercise scratch padding.
    reference = _runtime(model, batch_sizes=(batch_size,))
    buffered = _runtime(model, batch_sizes=(batch_size,))
    original_decode = buffered.decode_step
    executor = install_buffered_decode(buffered)
    try:
        assert buffered.decode_step([]) == {}
        assert executor.execution_info()["metadata_uploads"] == 0
        assert executor.execution_info()["buckets"] == []
        expected = [_add(reference, f"r{row}", 3 + row % 3) for row in range(batch_size)]
        actual = [_add(buffered, f"r{row}", 3 + row % 3) for row in range(batch_size)]
        assert reference.step() == buffered.step()  # Prefill, one output each.
        width = max(len(request.block_ids) for request in actual)
        executor.prepare(batch_size, width, warmup=1)
        pointers = _ptrs(executor)
        expected_samples, actual_samples = _observe_sampling(reference), _observe_sampling(buffered)
        full = list(range(batch_size))
        phases = [full, list(reversed(full))[:max(1, batch_size // 2)],
                  full[1:] + full[:1], list(reversed(full))]
        for order in phases:
            inactive = [page for row, request in enumerate(actual) if row not in order
                        for page in request.block_ids]
            before_key = buffered.cache.key[:, inactive].clone()
            before_value = buffered.cache.value[:, inactive].clone()
            before = executor.execution_info()
            expected_samples.clear()
            actual_samples.clear()
            result = reference.decode_step([expected[row] for row in order])
            assert buffered.decode_step([actual[row] for row in order]) == result
            assert len(expected_samples) == len(actual_samples) == 1
            assert actual_samples[0][0] == expected_samples[0][0]
            torch.testing.assert_close(actual_samples[0][1], expected_samples[0][1],
                                       atol=0.01, rtol=0.01)
            assert torch.equal(actual_samples[0][1].argmax(-1),
                               expected_samples[0][1].argmax(-1))
            _assert_requests_and_kv(reference, buffered, expected, actual)
            torch.testing.assert_close(buffered.cache.key[:, inactive], before_key,
                                       atol=0, rtol=0)
            torch.testing.assert_close(buffered.cache.value[:, inactive], before_value,
                                       atol=0, rtol=0)
            info = executor.execution_info()
            assert info["metadata_uploads"] == before["metadata_uploads"] + 1
            assert info["steps"] == before["steps"] + 1
            assert all(row["pinned"] for row in info["buckets"])
            current = _ptrs(executor)
            assert all(current[shape] == pointer for shape, pointer in pointers.items())
            pointers = current
            scratch = set(buffered.graph_padding_block_ids)
            assert all(scratch.isdisjoint(request.block_ids) for request in actual)
    finally:
        executor.close()
    assert buffered.decode_step == original_decode
    assert not hasattr(buffered, "_dandan_buffered_decode")
    assert buffered.decode_step([]) == {}
    executor.close()  # Idempotent shutdown.
    # Captured bucket views must keep the slab storage alive after workspaces
    # are cleared; the restored engine must still replay those existing graphs.
    assert reference.step() == buffered.step()
    _assert_requests_and_kv(reference, buffered, expected, actual)


@torch.inference_mode()
def test_page_growth_changes_bucket_and_reuses_old_slab(graph_cuda):
    model = _model()
    reference, buffered = _runtime(model, batch_sizes=(1,)), _runtime(model, batch_sizes=(1,))
    with install_buffered_decode(buffered) as executor:
        left, right = _add(reference, "long", 2), _add(buffered, "long", 2)
        assert reference.step() == buffered.step()
        first_bucket = executor.prepare(1, 1, warmup=1)
        first_ptrs = _ptrs(executor)[(1, 1)]
        for _ in range(3):
            assert reference.decode_step([left]) == buffered.decode_step([right])
        assert (1, 2) in _ptrs(executor)
        assert _ptrs(executor)[(1, 1)] == first_ptrs
        assert executor.prepare(1, 1) is first_bucket
        _assert_requests_and_kv(reference, buffered, [left], [right])
        # A newly admitted short request selects the old width-one bucket.
        # The long request remains live while its row is replaced.
        new_left, new_right = _add(reference, "short", 1), _add(buffered, "short", 1)
        assert reference.step() == buffered.step()
        old_key = buffered.cache.key[:, right.block_ids].clone()
        old_value = buffered.cache.value[:, right.block_ids].clone()
        before = executor.execution_info()["metadata_uploads"]
        assert reference.decode_step([new_left]) == buffered.decode_step([new_right])
        assert _ptrs(executor)[(1, 1)] == first_ptrs
        assert executor.execution_info()["metadata_uploads"] == before + 1
        torch.testing.assert_close(buffered.cache.key[:, right.block_ids], old_key, atol=0, rtol=0)
        torch.testing.assert_close(buffered.cache.value[:, right.block_ids], old_value, atol=0, rtol=0)
        _assert_requests_and_kv(reference, buffered, [left, new_left], [right, new_right])


@torch.inference_mode()
def test_eos_length_release_and_new_request_page_reuse(graph_cuda):
    runtime = _runtime(_model(), batch_sizes=(1, 2))
    runtime.eos_token_ids = {13}

    def forced_sample(logits, requests):
        return torch.tensor([13 if request.request_id == "eos" and request.generated_tokens else 7
                             for request in requests], device=logits.device, dtype=torch.long)

    runtime._sample = forced_sample
    with install_buffered_decode(runtime) as executor:
        eos = _add(runtime, "eos", 3, outputs=5)
        length = _add(runtime, "length", 5, outputs=2)
        runtime.step()
        released_pages = set(eos.block_ids + length.block_ids)
        runtime.step()
        assert eos.state is length.state is RequestState.FINISHED
        assert eos.finished_reason == "stop" and length.finished_reason == "length"
        assert all(runtime.scheduler.block_manager.refcount(page) == 0 for page in released_pages)
        scratch = set(runtime.graph_padding_block_ids)
        assert runtime.scheduler.block_manager.used_blocks == len(scratch)
        fresh = _add(runtime, "fresh", 3, outputs=2)
        runtime.step()
        assert released_pages.intersection(fresh.block_ids)
        assert scratch.isdisjoint(fresh.block_ids)
        before = executor.execution_info()["metadata_uploads"]
        runtime.step()
        assert fresh.state is RequestState.FINISHED
        assert fresh.output_token_ids == [7, 7]
        assert executor.execution_info()["metadata_uploads"] == before + 1
        assert runtime.scheduler.block_manager.used_blocks == len(scratch)


@torch.inference_mode()
def test_mixed_sampling_parameters_keep_original_sampler(graph_cuda):
    model = _model()
    reference, buffered = _runtime(model), _runtime(model)
    with install_buffered_decode(buffered):
        expected = [_add(reference, "greedy", temperature=0.0),
                    _add(reference, "random", temperature=0.7, top_k=5, top_p=0.9)]
        actual = [_add(buffered, "greedy", temperature=0.0),
                  _add(buffered, "random", temperature=0.7, top_k=5, top_p=0.9)]
        torch.manual_seed(211)
        first = reference.step()
        torch.manual_seed(211)
        assert buffered.step() == first
        expected_calls, actual_calls = _observe_sampling(reference), _observe_sampling(buffered)
        torch.manual_seed(223)
        chosen = reference.decode_step(list(reversed(expected)))
        torch.manual_seed(223)
        assert buffered.decode_step(list(reversed(actual))) == chosen
        assert len(expected_calls) == len(actual_calls) == 1
        assert actual_calls[0][0] == ("random", "greedy")
        torch.testing.assert_close(actual_calls[0][1], expected_calls[0][1], atol=0.01, rtol=0.01)
        assert [(r.temperature, r.top_k, r.top_p) for r in actual] == [(0.0, 0, 1.0), (0.7, 5, 0.9)]
        _assert_requests_and_kv(reference, buffered, expected, actual)


@torch.inference_mode()
def test_nondefault_installation_stream_and_cross_stream_rejection(graph_cuda):
    model = _model()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    executor = None
    try:
        with torch.cuda.stream(stream):
            runtime = _runtime(model, batch_sizes=(1,))
            executor = install_buffered_decode(runtime)
            request = _add(runtime, "stream")
            runtime.step()
            executor.prepare(1, 1, warmup=1)
            runtime.decode_step([request])
            assert executor.execution_info()["metadata_uploads"] == 1
        other = torch.cuda.Stream()
        with torch.cuda.stream(other):
            with pytest.raises(ValueError, match="stream"):
                runtime.decode_step([request])
        assert executor.execution_info()["metadata_uploads"] == 1
        with torch.cuda.stream(stream):
            runtime.decode_step([request])
            assert executor.execution_info()["metadata_uploads"] == 2
    finally:
        if executor is not None:
            executor.close()


@torch.inference_mode()
def test_reject_already_captured_and_external_capture_after_install(graph_cuda):
    model = _model()
    captured_runtime = _runtime(model, batch_sizes=(1,))
    captured_runtime.graph_manager.capture(
        1, 1, padding_block_ids=captured_runtime.graph_padding_block_ids, warmup=1
    )
    original = captured_runtime.decode_step
    with pytest.raises(ValueError, match="before capturing"):
        install_buffered_decode(captured_runtime)
    assert captured_runtime.decode_step == original
    runtime = _runtime(model, batch_sizes=(1,))
    with install_buffered_decode(runtime) as executor:
        external = runtime.graph_manager.capture(
            1, 2, padding_block_ids=runtime.graph_padding_block_ids, warmup=1
        )
        pointers = tuple(tensor.data_ptr() for tensor in
                         (external.input_ids, external.positions, external.block_tables))
        with pytest.raises(ValueError, match="externally captured"):
            executor.prepare(1, 2)
        assert tuple(tensor.data_ptr() for tensor in
                     (external.input_ids, external.positions, external.block_tables)) == pointers
        assert executor.execution_info()["buckets"] == []


@torch.inference_mode()
def test_capture_failure_restores_bucket_and_can_retry(graph_cuda, monkeypatch):
    runtime = _runtime(_model(), batch_sizes=(1,))
    bucket = runtime.graph_manager.get(
        1, 1, padding_block_ids=runtime.graph_padding_block_ids
    )
    original_fields = (bucket.input_ids, bucket.positions, bucket.block_tables, bucket.logits)
    original_capture = bucket.capture
    attempts = []

    def fail_once(*, warmup=3):
        attempts.append(1)
        if len(attempts) == 1:
            # Workspace binding has occurred; emulate a capture wrapper that
            # enqueued work and left partial output/Graph state before failing.
            bucket.input_ids.fill_(9)
            bucket.logits = torch.empty((1, 71), device="cuda", dtype=torch.float16)
            bucket.graph = object()
            raise RuntimeError("injected first capture failure")
        return original_capture(warmup=warmup)

    monkeypatch.setattr(bucket, "capture", fail_once)
    with install_buffered_decode(runtime) as executor:
        with pytest.raises(RuntimeError, match="injected first capture failure"):
            executor.prepare(1, 1, warmup=1)
        restored = (bucket.input_ids, bucket.positions, bucket.block_tables, bucket.logits)
        assert all(current is original for current, original in zip(restored, original_fields))
        assert bucket.graph is None and not bucket.captured
        assert runtime.graph_manager.buckets[bucket.shape] is bucket
        assert executor.execution_info()["buckets"] == []
        assert executor.execution_info()["metadata_uploads"] == 0
        assert executor.prepare(1, 1, warmup=1) is bucket
        assert bucket.captured and len(attempts) == 2
        request = _add(runtime, "retry", outputs=2)
        runtime.step()
        runtime.step()
        assert request.state is RequestState.FINISHED
        assert executor.execution_info()["metadata_uploads"] == 1


@pytest.mark.parametrize("invalid_page", [-1, 128], ids=["negative", "out_of_range"])
@torch.inference_mode()
def test_invalid_initial_scratch_rejected_before_capture(graph_cuda, invalid_page):
    runtime = _runtime(_model(), batch_sizes=(1,))
    original_decode = runtime.decode_step
    before_key, before_value = runtime.cache.key.clone(), runtime.cache.value.clone()
    runtime.graph_padding_block_ids[0] = invalid_page
    with pytest.raises(ValueError, match="scratch|padding|page"):
        install_buffered_decode(runtime)
    assert runtime.decode_step == original_decode
    assert not hasattr(runtime, "_dandan_buffered_decode")
    assert runtime.graph_manager.buckets == {}
    torch.testing.assert_close(runtime.cache.key, before_key, atol=0, rtol=0)
    torch.testing.assert_close(runtime.cache.value, before_value, atol=0, rtol=0)


@torch.inference_mode()
def test_scratch_requires_retained_owner_and_cannot_be_request_pages(graph_cuda):
    model = _model()
    unallocated = _runtime(model, batch_sizes=(1,))
    unallocated.graph_padding_block_ids = [125, 126, 127]
    with pytest.raises(ValueError, match="scratch|padding|owner|reserv"):
        install_buffered_decode(unallocated)
    assert unallocated.graph_manager.buckets == {}
    assert not hasattr(unallocated, "_dandan_buffered_decode")

    active = _runtime(model, batch_sizes=(1,))
    request = _add(active, "active_scratch", length=9)
    active.step()
    assert request.state is RequestState.RUNNING and len(request.block_ids) == 3
    before_key, before_value = active.cache.key.clone(), active.cache.value.clone()
    active.graph_padding_block_ids = list(request.block_ids)
    with pytest.raises(ValueError, match="scratch|padding|owner|request"):
        install_buffered_decode(active, scratch_owner=request.request_id)
    assert active.graph_manager.buckets == {}
    assert not hasattr(active, "_dandan_buffered_decode")
    torch.testing.assert_close(active.cache.key, before_key, atol=0, rtol=0)
    torch.testing.assert_close(active.cache.value, before_value, atol=0, rtol=0)


@torch.inference_mode()
def test_custom_retained_scratch_owner_can_capture_and_generate(graph_cuda):
    runtime = _runtime(_model(), batch_sizes=(1,))
    manager = runtime.scheduler.block_manager
    default_pages = tuple(runtime.graph_padding_block_ids)
    owner = "buffered-test-padding"
    custom_pages = manager.allocate(owner, 3 * runtime.cache.shape.block_size)
    runtime.graph_padding_block_ids = list(custom_pages)
    assert set(custom_pages).isdisjoint(default_pages)
    with install_buffered_decode(runtime, scratch_owner=owner) as executor:
        executor.prepare(1, 1, warmup=1)
        request = _add(runtime, "custom", outputs=2)
        runtime.step()
        runtime.step()
        assert request.state is RequestState.FINISHED
        assert executor.execution_info()["metadata_uploads"] == 1
        assert all(manager.refcount(page) == 1 for page in custom_pages)
        assert manager.used_blocks == len(default_pages) + len(custom_pages)


@torch.inference_mode()
def test_sync_failure_preserves_capture_error_and_close_restores_runtime(graph_cuda, monkeypatch):
    runtime = _runtime(_model(), batch_sizes=(1,))
    bucket = runtime.graph_manager.get(
        1, 1, padding_block_ids=runtime.graph_padding_block_ids
    )
    old_fields = (bucket.input_ids, bucket.positions, bucket.block_tables, bucket.logits)
    original_decode = runtime.decode_step
    executor = install_buffered_decode(runtime)
    actual_stream = executor.stream

    class FailedSyncStream:
        cuda_stream = actual_stream.cuda_stream

        def synchronize(self):
            raise RuntimeError("injected secondary synchronize error")

    def fail_capture(*, warmup=3):
        # No GPU work is queued by this injected failure, allowing a CPU stub
        # to model a failed fence without leaving genuine GPU writes in flight.
        bucket.logits = torch.empty((1, 71), device="cuda", dtype=torch.float16)
        bucket.graph = object()
        raise RuntimeError("injected primary capture error")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(executor, "stream", FailedSyncStream())
            patch.setattr(executor, "_check_stream", lambda: None)
            patch.setattr(bucket, "capture", fail_capture)
            with pytest.raises(RuntimeError, match="injected primary capture error") as capture_error:
                executor.prepare(1, 1, warmup=1)
            assert any("injected secondary synchronize error" in note
                       for note in getattr(capture_error.value, "__notes__", ()))
            current_fields = (bucket.input_ids, bucket.positions, bucket.block_tables, bucket.logits)
            assert all(current is old for current, old in zip(current_fields, old_fields))
            assert bucket.graph is None and not bucket.captured
            assert runtime.graph_manager.buckets[bucket.shape] is bucket
            assert executor.execution_info()["buckets"] == []
            with pytest.raises(RuntimeError, match="injected secondary synchronize error"):
                executor.close()
            assert executor.closed
            assert runtime.decode_step == original_decode
            assert not hasattr(runtime, "_dandan_buffered_decode")
            assert runtime.decode_step([]) == {}
    finally:
        actual_stream.synchronize()
        if not executor.closed:
            executor.stream = actual_stream
            executor.close()

    # Context-manager cleanup must also preserve a body exception rather than
    # replace it with the secondary close/fence failure.
    context_executor = install_buffered_decode(runtime)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(context_executor, "stream", FailedSyncStream())
            with pytest.raises(RuntimeError, match="injected primary body error") as body_error:
                with context_executor:
                    raise RuntimeError("injected primary body error")
            assert any("injected secondary synchronize error" in note
                       for note in getattr(body_error.value, "__notes__", ()))
            assert context_executor.closed
            assert runtime.decode_step == original_decode
            assert not hasattr(runtime, "_dandan_buffered_decode")
    finally:
        if not context_executor.closed:
            context_executor.stream = actual_stream
            context_executor.close()


@torch.inference_mode()
def test_invalid_request_does_not_capture_upload_or_write_scratch(graph_cuda):
    runtime = _runtime(_model(), batch_sizes=(1,))
    with install_buffered_decode(runtime) as executor:
        request = _add(runtime, "invalid")
        runtime.step()  # Prefill uses no decode Graph bucket.
        before_key, before_value = runtime.cache.key.clone(), runtime.cache.value.clone()
        request.output_token_ids[-1] = runtime.model.config.vocab_size
        with pytest.raises(ValueError, match="vocabulary"):
            runtime.decode_step([request])
        assert runtime.graph_manager.buckets == {}
        assert executor.execution_info()["buckets"] == []
        assert executor.execution_info()["metadata_uploads"] == 0
        assert not executor.busy
        torch.testing.assert_close(runtime.cache.key, before_key, atol=0, rtol=0)
        torch.testing.assert_close(runtime.cache.value, before_value, atol=0, rtol=0)


@torch.inference_mode()
def test_tensor_scratch_inplace_mutation_is_rejected_before_capture(graph_cuda):
    runtime = _runtime(_model(), batch_sizes=(1,))
    # A normal tensor supplies a version counter even though decode executes
    # under inference_mode. Inference tensors intentionally have no counter.
    with torch.inference_mode(False):
        padding = torch.tensor(runtime.graph_padding_block_ids, dtype=torch.long)
    runtime.graph_padding_block_ids = padding
    with install_buffered_decode(runtime) as executor:
        request = _add(runtime, "padding")
        runtime.step()
        with torch.inference_mode(False):
            padding[0] = runtime.cache.shape.num_blocks - 1
        with pytest.raises(ValueError, match="scratch configuration changed"):
            runtime.decode_step([request])
        with pytest.raises(ValueError, match="scratch configuration changed"):
            executor.prepare(1, 1)
        assert runtime.graph_manager.buckets == {}
        assert executor.execution_info()["metadata_uploads"] == 0


@torch.inference_mode()
def test_sampling_failure_aborts_decode_and_reserved_prefill(graph_cuda, monkeypatch):
    runtime = _runtime(_model(), batch_sizes=(1, 2))
    with install_buffered_decode(runtime) as executor:
        active = _add(runtime, "active")
        runtime.step()
        waiting = _add(runtime, "waiting", 2)
        original_sample = runtime._sample

        def fail(logits, requests):
            raise RuntimeError("injected sampling failure after graph replay")

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_sample", fail)
            with pytest.raises(RuntimeError, match="injected sampling failure"):
                runtime.step()
        assert runtime._sample == original_sample
        assert active.state is waiting.state is RequestState.ABORTED
        assert not runtime.scheduler.running and not runtime.scheduler.waiting
        assert runtime.scheduler.block_manager.used_blocks == len(runtime.graph_padding_block_ids)
        assert executor.execution_info()["metadata_uploads"] == 1
        assert executor.execution_info()["steps"] == 0
        assert not executor.busy
        # Released pages can immediately serve a fresh request, proving the
        # error fence completed before the engine recycled those pages.
        fresh = _add(runtime, "recovery", 3, outputs=2)
        runtime.step()
        runtime.step()
        assert fresh.state is RequestState.FINISHED
        assert runtime.scheduler.block_manager.used_blocks == len(runtime.graph_padding_block_ids)


@torch.inference_mode()
def test_chunked_prefill_shares_readonly_prefix_during_buffered_decode(graph_cuda):
    model = _model()
    reference = _runtime(model, enable_prefix_cache=True, prefill_chunk_size=2)
    buffered = _runtime(model, enable_prefix_cache=True, prefill_chunk_size=2)
    prefix = [1, 2, 3, 4]
    with install_buffered_decode(buffered):
        primes = []
        for runtime in (reference, buffered):
            prime = GenerationRequest("prime", list(prefix), max_new_tokens=1)
            runtime.scheduler.add_request(prime)
            primes.append(prime)
        for _ in range(2):
            assert reference.step() == buffered.step()
        assert all(prime.state is RequestState.FINISHED for prime in primes)
        entries = [runtime.prefix_cache.lookup(prefix) for runtime in (reference, buffered)]
        assert all(entry is not None and entry.next_logits is not None for entry in entries)
        shared_snapshots = [(runtime.cache.key[:, list(entry.block_ids)].clone(),
                             runtime.cache.value[:, list(entry.block_ids)].clone())
                            for runtime, entry in zip((reference, buffered), entries)]
        pairs, decode_before_other_prefill, logits_calls = [], [False, False], [[], []]
        for index, runtime in enumerate((reference, buffered)):
            short = GenerationRequest("short", prefix + [21], max_new_tokens=4)
            long = GenerationRequest("long", prefix + [22, 23, 24], max_new_tokens=3)
            for request in (short, long):
                runtime.scheduler.add_request(request)
            pairs.append([short, long])
            original_sample = runtime._sample

            def observe(logits, requests, *, slot=index, children=(short, long), sample=original_sample):
                if any(request is children[0] and request.generated_tokens > 0 for request in requests):
                    if children[1].generated_tokens == 0:
                        decode_before_other_prefill[slot] = True
                logits_calls[slot].append((tuple(request.request_id for request in requests), logits.clone()))
                return sample(logits, requests)

            runtime._sample = observe
        saw_partial_suffix = False
        for _ in range(8):
            if not reference.scheduler.waiting and not reference.scheduler.running:
                break
            logits_calls[0].clear()
            logits_calls[1].clear()
            assert reference.step() == buffered.step()
            assert len(logits_calls[0]) == len(logits_calls[1])
            for expected_call, actual_call in zip(logits_calls[0], logits_calls[1]):
                assert actual_call[0] == expected_call[0]
                torch.testing.assert_close(actual_call[1], expected_call[1], atol=0.01, rtol=0.01)
            _assert_requests_and_kv(reference, buffered, pairs[0], pairs[1])
            for runtime, children, entry, (saved_key, saved_value) in zip(
                    (reference, buffered), pairs, entries, shared_snapshots):
                assert all(request.prefix_len == len(prefix) for request in children)
                assert all(tuple(request.prefix_block_ids) == entry.block_ids for request in children)
                torch.testing.assert_close(runtime.cache.key[:, list(entry.block_ids)], saved_key, atol=0, rtol=0)
                torch.testing.assert_close(runtime.cache.value[:, list(entry.block_ids)], saved_value, atol=0, rtol=0)
                if 0 < children[1].prefill_cursor < children[1].prompt_len:
                    saw_partial_suffix = True
                for request in children:
                    if request.is_finished or not request.output_token_ids:
                        continue
                    position = request.total_len - 1
                    write_page = request.block_ids[position // runtime.cache.shape.block_size]
                    assert write_page not in entry.block_ids
                    assert all(write_page not in other.block_ids for other in children if other is not request)
        assert saw_partial_suffix and all(decode_before_other_prefill)
        assert not reference.scheduler.running and not buffered.scheduler.running
        for runtime, children, entry in zip((reference, buffered), pairs, entries):
            assert all(request.state is RequestState.FINISHED for request in children)
            assert [request.generated_tokens for request in children] == [4, 3]
            manager = runtime.scheduler.block_manager
            assert manager.used_blocks == len(runtime.graph_padding_block_ids) + len(entry.block_ids)
            assert all(manager.refcount(page) == 1 for page in entry.block_ids)
            runtime.prefix_cache.clear()
            assert manager.used_blocks == len(runtime.graph_padding_block_ids)

