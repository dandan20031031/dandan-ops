"""Load the prebuilt extension without JIT compilation."""
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import torch

_loaded = False


def _extension_candidates(directory, suffixes=None):
    """Find exact platform/Python tagged binaries; ignore untagged stale files."""
    suffixes = EXTENSION_SUFFIXES if suffixes is None else suffixes
    generic = {".pyd", ".so", ".abi3.so"}
    return sorted({directory / f"_C{suffix}" for suffix in suffixes
                   if suffix not in generic and (directory / f"_C{suffix}").is_file()})

def load_extension(required=False):
    global _loaded
    if _loaded:
        return True
    if torch.version.cuda is None:
        if required:
            raise ImportError("dandan_ops was built without CUDA support")
        return False
    # The binary is intentionally tied to the Python/PyTorch ABI that loaded it.
    # A stale .pyd from another venv must not be silently loaded.
    here = Path(__file__).resolve().parent
    libs = _extension_candidates(here)
    if not libs:
        if required:
            raise ImportError("dandan_ops CUDA extension is unavailable; run pip install -e .")
        return False
    if len(libs) != 1:
        raise ImportError(f"ambiguous dandan_ops extensions: {libs}")
    torch.ops.load_library(str(libs[0]))
    try:
        contract = torch.ops.dandan_ops.w4a16_contract_version()
    except AttributeError as exc:
        raise ImportError("stale dandan_ops binary: rebuild for W4A16 contract version 2") from exc
    if contract != 2:
        raise ImportError(f"incompatible W4A16 contract {contract}; rebuild dandan_ops")
    _loaded = True
    return True

load_extension()
