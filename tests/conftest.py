"""Keep engine Graph cases independent of Python cyclic-GC timing."""
import gc

import pytest


@pytest.fixture(autouse=True)
def isolate_engine_graph_lifetimes(request):
    if "graph_cuda" not in request.fixturenames:
        yield
        return

    import torch

    # A runtime contains cycles through scheduler callbacks. Collect old
    # Graph owners outside capture, and avoid CUDA-backed destructors running
    # from automatic GC in the middle of a later engine capture.
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            if enabled:
                gc.enable()
            gc.collect()
