"""Verify the delivered wheel in a fresh process, then remove smoke files."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

library = Path(sys.argv[1]).resolve()
wheel = library / "build/wheels/dandan_ops-0.1.0-cp313-cp313-win_amd64.whl"
files = ["dandan_ops/integrations/myvllm_decode.py",
         "dandan_ops/integrations/myvllm.py",
         "dandan_ops/profiles/qwen3_sm120_cu128.json"]
with zipfile.ZipFile(wheel) as archive:
    for name in files:
        assert archive.read(name) == (library / name).read_bytes(), name
    profile = json.loads(archive.read(files[-1]))
    routes = sum(len(value) for value in profile["routes"].values())
    assert routes == 98
    metadata = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))).decode()
    assert "Provides-Extra: myvllm" in metadata
    assert any("numpy" in line and 'extra == "myvllm"' in line for line in metadata.splitlines())
    temporary = (library / "build/tmp").resolve()
    extracted = Path(tempfile.mkdtemp(prefix="decode-wheel-smoke-", dir=temporary)).resolve()
    try:
        archive.extractall(extracted)
        smoke = r'''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch
import dandan_ops
from dandan_ops._extension import load_extension
from dandan_ops.integrations.myvllm_decode import install_buffered_decode,fill_decode_metadata
from types import SimpleNamespace
assert Path(dandan_ops.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert "mini_engine" not in sys.modules
load_extension(required=True)
assert torch.ops.dandan_ops.w4a16_contract_version() == 2
array=np.empty(8,dtype=np.int64)
request=SimpleNamespace(output_token_ids=[7],total_len=2,block_ids=[1],is_finished=False)
fill_decode_metadata(array,[request],batch_size=2,max_blocks=2,padding=(6,7),block_size=4,num_blocks=8)
assert array.tolist()==[7,0,1,0,1,-1,7,7]
print("isolated wheel import, extension contract v2, CPU metadata and no engine import: passed")
'''
        result = subprocess.run([sys.executable,"-I","-c",smoke,str(extracted)],
                                capture_output=True,text=True,check=True)
        report = {"wheel":str(wheel),"wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "matching_sources":files,"routes":routes,"numpy_extra":True,
                  "isolated_smoke":result.stdout.strip()}
        (library / "build/wheel_decode_buffers_smoke.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2))
    finally:
        # Resolve and verify containment before recursive cleanup on Windows.
        if not extracted.is_relative_to(temporary):
            raise RuntimeError("unexpected smoke path outside build/tmp")
        shutil.rmtree(extracted)
