param([string]$EngineRoot = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'myvllm'))
$ErrorActionPreference = 'Stop'
$opsRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $EngineRoot '.venv\Scripts\python.exe'
$cuda = Join-Path $EngineRoot '.cuda-12.8\Library'
if (-not (Test-Path -LiteralPath $python)) { throw "Engine Python missing: $python" }
if (-not (Test-Path -LiteralPath (Join-Path $cuda 'bin\nvcc.exe'))) { throw "CUDA 12.8 nvcc missing: $cuda" }
# All temporary/compiler/cache output stays under the operator library.
$buildTmp = Join-Path $opsRoot 'build\tmp'
New-Item -ItemType Directory -Force -Path $buildTmp | Out-Null
$env:TEMP=$buildTmp; $env:TMP=$buildTmp
$env:DANDAN_CUDA_HOME=$cuda
$env:CUDA_HOME=$cuda; $env:CUDA_PATH=$cuda
$env:VSLANG='1033'
$env:MAX_JOBS='2'
$env:PATH="$(Join-Path $EngineRoot '.venv\Scripts');$(Join-Path $cuda 'bin');$env:PATH"
# setup.py initializes the full VS2022 environment, not just -ccbin.
& $python -m pip install -e $opsRoot --no-build-isolation --no-deps
if ($LASTEXITCODE -ne 0) { throw 'dandan_ops build/install failed' }
& $python -c "import torch,dandan_ops; print('torch:',torch.__version__); print('W4A16 contract:',torch.ops.dandan_ops.w4a16_contract_version())"
if ($LASTEXITCODE -ne 0) { throw 'dandan_ops import failed' }
