# Contributing

Issues and pull requests are welcome. For a bug, provide a minimal example,
GPU/SM, OS, Python/PyTorch/CUDA versions, input shape/dtype/stride, selected
backend and the error or numerical mismatch. Avoid including model weights,
credentials or machine-specific private files.

Install the `dev` extra and run `python -m pytest -q`. CPU CI checks contracts
and fallbacks; CUDA changes also need tests on the affected GPU. myvllm
integration tests are optional and use `MYVLLM_ROOT` with a compatible local
checkout and prebuilt engine extension.

For a kernel change, compare against the same numerical contract, including
FP16 dequantization rounding for W4A16. Cover relevant tails, alignment,
streams and Graph lifetime. Record benchmark commands, timing boundaries,
software/device versions and raw samples. Keep operator microbenchmarks,
full forward and generation-loop measurements distinct.

Do not add a runtime dependency on the original experiment tree. Preserve
source attribution and license notices. Contributions are made under this
repository's MIT license.

## Build and check a wheel

Use the validated Windows toolchain (or configure a matching CUDA Toolkit):

```powershell
./scripts/build_for_myvllm.ps1 -EngineRoot ../myvllm
../myvllm/.venv/Scripts/python.exe -m pytest -q
../myvllm/.venv/Scripts/python.exe -m pip wheel . --no-build-isolation --no-deps --wheel-dir build/wheels
../myvllm/.venv/Scripts/python.exe scripts/verify_gpu_wheel.py build/wheels/dandan_ops-0.1.0-cp313-cp313-win_amd64.whl
```

Adjust the wheel filename to the Python/platform tag produced by the build.
The last command installs into a temporary target, checks a fresh-process import
and GPU inference, then cleans that target. Its result is written to
`build/wheel_gpu_smoke.json`. Keep one verifier for the whole library; repeatable
regression tests belong in `tests/`, not one-off root-level scripts.
