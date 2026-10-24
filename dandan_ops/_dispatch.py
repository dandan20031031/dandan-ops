"""Offline measured routes; no timing, value inspection, or file I/O in forward."""
from functools import lru_cache
import json
from pathlib import Path

import torch

DISPATCH_REVISION = 1
_path = Path(__file__).with_name("profiles") / "qwen3_sm120_cu128.json"
_profile = json.loads(_path.read_text(encoding="utf-8")) if _path.is_file() else {}


@lru_cache(maxsize=None)
def _device_signature(device_index):
    return {"gpu": torch.cuda.get_device_name(device_index),
            "sm": list(torch.cuda.get_device_capability(device_index)),
            # TorchVersion overloads equality with packaging.version parsing;
            # use plain strings in the per-call dictionary comparison.
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda}


def measured_route(family, m, n, k, device):
    if _profile.get("dispatch_revision") != DISPATCH_REVISION:
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if _profile.get("environment") != _device_signature(index):
        return None
    route = _profile.get("routes", {}).get(family, {}).get(f"{m}x{n}x{k}")
    allowed = {"dense": ("torch", "cuda", "tma"),
               "w4a16": ("w4a16_small", "w4a16_gemv", "w4a16_tensorcore")}
    return route if route in allowed.get(family, ()) else None
