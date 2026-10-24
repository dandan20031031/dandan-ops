"""Optional graph integration against an existing local myvllm checkout.

No engine source files are changed and no extension is built.  The benchmark
helper loads an already-built engine extension into this process.  Each test
uses its own KV pools, including private capture/padding pages.
"""

import os
from pathlib import Path
import sys

import pytest
import torch

import dandan_ops as ops
from dandan_ops.integrations.myvllm import install_int4_projections


_LIBRARY_ROOT = Path(ops.__file__).resolve().parents[1]
_ENGINE_ROOT = Path(os.environ.get("MYVLLM_ROOT", _LIBRARY_ROOT.parent / "myvllm"))
if not (_ENGINE_ROOT / "mini_engine").is_dir():
    pytest.skip("optional myvllm checkout not found", allow_module_level=True)
sys.path.insert(0, str(_ENGINE_ROOT))
sys.path.insert(0, str(_LIBRARY_ROOT))

from mini_engine.config import QuantConfig
from mini_engine.engine.cuda_graph import DecodeGraphBucket
from mini_engine.model import NativeQwen3, Qwen3Config


@pytest.fixture
def graph_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("dandan inference kernels require SM80 or newer")

    from dandan_ops._extension import load_extension
    import mini_engine.kernels.cuda_extension as engine_extension
    from benchmarks.bench_qwen3_graph import load_prebuilt_extension

    # Register the original callable for restoration before the helper changes
    # the process-local loader.  Never invoke the original build-on-demand path.
    monkeypatch.setattr(engine_extension, "load_cuda_extension",
                        engine_extension.load_cuda_extension)
    try:
        load_prebuilt_extension(_ENGINE_ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"optional prebuilt myvllm extension not found: {exc}")
    load_extension(required=True)


def _assert_all_live_kv_equal(actual, expected, live_pages):
    # Compare entire physical pages, including unwritten future positions and
    # inactive rows.  A padding write into a future live page must also fail.
    torch.testing.assert_close(actual.key.index_select(1, live_pages),
                               expected.key.index_select(1, live_pages),
                               atol=0.005, rtol=0.01)
    torch.testing.assert_close(actual.value.index_select(1, live_pages),
                               expected.value.index_select(1, live_pages),
                               atol=0.005, rtol=0.01)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("quantized", [False, True], ids=["fp16", "int4"])
@torch.inference_mode()
def test_myvllm_decode_graph_preserves_live_kv(graph_cuda, batch_size, quantized):
    torch.manual_seed(107)
    # D=128 with true GQA and two layers; every projection K is compatible
    # with the adapter's source group_size=128 and target group_size=64.
    cfg = Qwen3Config(71, 128, 256, 2, 4, 2, 128,
                      max_position_embeddings=64)
    model = NativeQwen3(cfg, prefill_backend="math", decode_backend="cuda",
                        data_backend="cuda", elementwise_backend="cuda")
    model.cuda().half().eval()
    if quantized:
        report = install_int4_projections(
            model, QuantConfig("int4", activation_dtype="fp16", group_size=128)
        )
        assert len(report["replaced_layers"]) == 14
        assert report["accelerated_quant_gemm"]

    block_size, max_blocks = 4, 3
    # Even-numbered pages are live, with noncontiguous pages in each request
    # even for B=1.  Reserved scratch pages follow the complete live range.
    request_pages = [[2 * (column * batch_size + row)
                      for column in range(max_blocks)]
                     for row in range(batch_size)]
    live_page_ids = [page for row in request_pages for page in row]
    scratch_count = max(batch_size, max_blocks)
    scratch_start = 2 * batch_size * max_blocks
    scratch_pages = list(range(scratch_start, scratch_start + scratch_count))
    assert set(live_page_ids).isdisjoint(scratch_pages)
    assert len(set(scratch_pages[:batch_size])) == batch_size
    num_blocks = scratch_pages[-1] + 1
    eager_cache = model.create_cache(num_blocks, block_size)
    graph_cache = model.create_cache(num_blocks, block_size)
    eager_cache.clear()
    graph_cache.clear()
    live_pages = torch.tensor(live_page_ids, device="cuda", dtype=torch.long)
    tables = torch.tensor(request_pages, device="cuda", dtype=torch.long)

    bucket = DecodeGraphBucket(model, graph_cache, batch_size, max_blocks,
                               padding_block_ids=scratch_pages)
    # Capture before prefill so accidental writes to any live page are visible.
    bucket.capture(warmup=2)
    assert bucket.block_tables[:, 0].tolist() == scratch_pages[:batch_size]
    torch.testing.assert_close(graph_cache.key.index_select(1, live_pages),
                               eager_cache.key.index_select(1, live_pages),
                               atol=0, rtol=0)
    torch.testing.assert_close(graph_cache.value.index_select(1, live_pages),
                               eager_cache.value.index_select(1, live_pages),
                               atol=0, rtol=0)
    buffer_ptrs = (bucket.input_ids.data_ptr(), bucket.positions.data_ptr(),
                   bucket.block_tables.data_ptr(), bucket.logits.data_ptr())

    prefix_lengths = [3 + row % 3 for row in range(batch_size)]
    positions_cpu = list(prefix_lengths)
    tokens = []
    for row, prefix_length in enumerate(prefix_lengths):
        prompt = (torch.arange(prefix_length, device="cuda", dtype=torch.long)
                  + 1 + 7 * row) % cfg.vocab_size
        expected = model(prompt, eager_cache, request_pages[row])
        actual = model(prompt, graph_cache, request_pages[row])
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        assert torch.equal(actual[-1].argmax(), expected[-1].argmax())
        tokens.append(expected[-1].argmax().reshape(1))
    tokens = torch.cat(tokens)
    _assert_all_live_kv_equal(graph_cache, eager_cache, live_pages)

    # For B=1 a nonempty partial batch cannot exist; retain the same three-step
    # schedule while the other sizes exercise inactive slots and reactivation.
    partial_count = max(1, batch_size // 2)
    for active_count in (batch_size, partial_count, batch_size):
        inactive_pages = torch.tensor(
            [page for row in request_pages[active_count:] for page in row],
            device="cuda", dtype=torch.long,
        )
        before_inactive_key = graph_cache.key.index_select(1, inactive_pages).clone()
        before_inactive_value = graph_cache.value.index_select(1, inactive_pages).clone()
        positions = torch.tensor(positions_cpu[:active_count],
                                 device="cuda", dtype=torch.long)
        expected = model.forward_decode_batch(
            tokens[:active_count], eager_cache, tables[:active_count], positions
        )
        actual = bucket.replay(tokens[:active_count], positions, tables[:active_count],
                               active_count=active_count)
        assert actual.shape == (active_count, cfg.vocab_size)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        assert buffer_ptrs == (bucket.input_ids.data_ptr(), bucket.positions.data_ptr(),
                               bucket.block_tables.data_ptr(), bucket.logits.data_ptr())
        assert bucket.block_tables[active_count:, 0].tolist() == scratch_pages[active_count:batch_size]
        _assert_all_live_kv_equal(graph_cache, eager_cache, live_pages)
        torch.testing.assert_close(graph_cache.key.index_select(1, inactive_pages),
                                   before_inactive_key, atol=0, rtol=0)
        torch.testing.assert_close(graph_cache.value.index_select(1, inactive_pages),
                                   before_inactive_value, atol=0, rtol=0)
        # Consume the graph-owned output before another replay overwrites it.
        tokens[:active_count] = actual.argmax(-1)
        for row in range(active_count):
            positions_cpu[row] += 1

