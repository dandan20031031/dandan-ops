"""Newer native metadata/sampling implementations must not be rebound."""
from types import SimpleNamespace
from types import ModuleType
import os
import sys

import pytest

from dandan_ops.integrations.myvllm_decode import install_buffered_decode


@pytest.mark.parametrize("capture_sampling", [True, False])
def test_native_graph_api_is_rejected_before_any_cuda_access(capture_sampling):
    runtime = SimpleNamespace(graph_manager=SimpleNamespace(capture_sampling=capture_sampling))
    with pytest.raises(ValueError, match="use its native runtime"):
        install_buffered_decode(runtime)
    assert not hasattr(runtime, "_dandan_buffered_decode")


@pytest.mark.parametrize("compatible", [True, False])
def test_prebuilt_loader_rejects_stale_engine_exports(tmp_path, monkeypatch, compatible):
    from benchmarks.bench_qwen3_graph import load_prebuilt_extension

    loader = ModuleType("mini_engine.kernels.cuda_extension")
    original = lambda: "original loader"
    loader.load_cuda_extension = original
    kernels = ModuleType("mini_engine.kernels")
    kernels.cuda_extension = loader
    engine = ModuleType("mini_engine")
    engine.kernels = kernels
    for name, module in (("mini_engine", engine), ("mini_engine.kernels", kernels),
                         ("mini_engine.kernels.cuda_extension", loader)):
        monkeypatch.setitem(sys.modules, name, module)
    bindings = tmp_path / "mini_engine/kernels/csrc/bindings.cpp"
    bindings.parent.mkdir(parents=True)
    bindings.write_text('m.def("paged_attention"); m.def("write_kv_cache_slots");')
    build = tmp_path / "build/torch_extensions/paged_attention"
    old = build / "newer-incompatible/miniqwen_paged_attention_stale.pyd"
    good = build / "older-compatible/miniqwen_paged_attention_good.pyd"
    for path in (old, good):
        path.parent.mkdir(parents=True)
        path.touch()
    os.utime(old, (200, 200))
    os.utime(good, (100, 100))
    stale_module = SimpleNamespace(paged_attention=object())
    good_module = SimpleNamespace(paged_attention=object(), write_kv_cache_slots=object())
    monkeypatch.setitem(sys.modules, old.stem, stale_module)
    monkeypatch.setitem(sys.modules, good.stem, good_module if compatible else stale_module)
    if compatible:
        assert load_prebuilt_extension(tmp_path) == good
        assert loader.load_cuda_extension() is good_module
    else:
        with pytest.raises(FileNotFoundError, match="write_kv_cache_slots"):
            load_prebuilt_extension(tmp_path)
        assert loader.load_cuda_extension is original
