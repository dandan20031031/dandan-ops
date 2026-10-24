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
