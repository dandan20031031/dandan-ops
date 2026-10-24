"""ABI discovery must work for platform tags without loading dummy binaries."""
from pathlib import Path

import pytest

from dandan_ops._extension import _extension_candidates


@pytest.mark.parametrize("suffixes,current,stale", [
    ([".cp313-win_amd64.pyd", ".pyd"], ".cp313-win_amd64.pyd", ".cp312-win_amd64.pyd"),
    ([".cpython-313-x86_64-linux-gnu.so", ".abi3.so", ".so"],
     ".cpython-313-x86_64-linux-gnu.so", ".cpython-312-x86_64-linux-gnu.so"),
])
def test_selects_exact_platform_and_python_tag(tmp_path, suffixes, current, stale):
    for suffix in (current, stale, ".so", ".pyd", ".abi3.so"):
        (tmp_path / f"_C{suffix}").touch()
    assert _extension_candidates(tmp_path, suffixes) == [tmp_path / f"_C{current}"]


def test_stale_and_untagged_binaries_do_not_enable_extension(tmp_path):
    for name in ("_C.cp312-win_amd64.pyd", "_C.pyd", "_C.so"):
        (tmp_path / name).touch()
    assert _extension_candidates(tmp_path, [".cp313-win_amd64.pyd", ".pyd"]) == []
