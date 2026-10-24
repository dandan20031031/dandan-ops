"""Newer native metadata/sampling implementations must not be rebound."""
from types import SimpleNamespace

import pytest

from dandan_ops.integrations.myvllm_decode import install_buffered_decode


@pytest.mark.parametrize("capture_sampling", [True, False])
def test_native_graph_api_is_rejected_before_any_cuda_access(capture_sampling):
    runtime = SimpleNamespace(graph_manager=SimpleNamespace(capture_sampling=capture_sampling))
    with pytest.raises(ValueError, match="use its native runtime"):
        install_buffered_decode(runtime)
    assert not hasattr(runtime, "_dandan_buffered_decode")
