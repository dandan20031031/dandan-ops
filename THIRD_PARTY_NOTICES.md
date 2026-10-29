# Sources and third-party dependencies

The CUDA kernels were extracted and adapted from the author's local
`cuda_code` experiments. The original experiment checkout is not required to
build or use this repository.

| Library file | Experiment source |
|---|---|
| `kernels/hgemm_pipeline.cuh` | `hgemm/hgemm_v8_sm120_mmasync.cu` |
| `kernels/hgemm_tma.cuh` | `hgemm/hgemm_v10_tma.cu` |
| `kernels/flash_tensorcore_inference.cuh` | `flashAttention/flashattention_v2_1.cuh` |
| `kernels/flashattention_v2_1.cuh` | Original D=64 attention experiment retained for reference |
| Other Norm/Softmax kernels | Corresponding versioned experiments, named in their filenames |
| `dandan_ops/csrc/elementwise.cu` | Author's `myvllm/mini_engine/kernels/csrc/elementwise.cu`; source SHA256 retained in header |

Low-precision RMSNorm, residual RMSNorm and SiLU multiplication were adapted
from the author's myvllm engine, with independent registration and boundary
checks. The engine checkout is not a build or runtime dependency. FP32 LayerNorm
was updated to use centered variance rather than the original sum-of-squares
subtraction; see [numeric contracts](docs/DEVELOPMENT.md#normsoftmax-与融合小算子).

Source provenance is preserved in kernel headers, this file and
[interface notes](docs/DEVELOPMENT.md). Public benchmark reports replace
machine-specific absolute paths with portable labels; measurements, output
tokens and recorded source/binary hashes are preserved.

This repository uses PyTorch and the NVIDIA CUDA Toolkit through their public
APIs. Those dependencies remain subject to their own licenses and are not
redistributed in the source repository. CUTLASS is not a build dependency.

FlashAttention refers to the attention algorithm and this project's inference
implementation; it does not imply an affiliation with the upstream
FlashAttention project. Qwen3 model weights, tokenizer files and the optional
myvllm engine are not included in this repository. Obtain them separately
under their respective licenses.
