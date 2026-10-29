import os
import sys
from pathlib import Path

# Respect explicit toolkit selection and also support conda/venv layouts.
env_root = Path(sys.prefix)
env_cuda = env_root / "Library"
env_scripts = env_root / ("Scripts" if os.name == "nt" else "bin")
requested_cuda = (os.environ.get("DANDAN_CUDA_HOME") or os.environ.get("CUDA_HOME")
                  or os.environ.get("CUDA_PATH"))
cuda_root = Path(requested_cuda) if requested_cuda else (env_cuda if os.name == "nt" else env_root)
if (cuda_root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")).is_file():
    os.environ["CUDA_HOME"] = str(cuda_root)
    os.environ["CUDA_PATH"] = str(cuda_root)
    os.environ["PATH"] = str(cuda_root / "bin") + os.pathsep + os.environ.get("PATH", "")
if env_scripts.is_dir():
    os.environ["PATH"] = str(env_scripts) + os.pathsep + os.environ.get("PATH", "")

# Initialize the complete VS2022 environment, including headers and linker.
# Merely changing -ccbin mixes VS18 headers into a VS2022 build.
if os.name == "nt":
    import subprocess
    installer = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if installer.is_file():
        location = subprocess.check_output([
            str(installer), "-version", "[17.0,18.0)", "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath",
        ], text=True).strip().splitlines()
        if location:
            vcvars = Path(location[0]) / "VC/Auxiliary/Build/vcvarsall.bat"
            command = f'call "{vcvars}" x64 >nul && set'
            result = subprocess.run(f'cmd /d /s /c "{command}"',
                                    stdout=subprocess.PIPE, check=True)
            for line in result.stdout.decode("mbcs", errors="replace").splitlines():
                key, separator, value = line.partition("=")
                if separator and key:
                    os.environ[key] = value
            os.environ["DISTUTILS_USE_SDK"] = "1"
            import shutil
            os.environ["NVCC_CCBIN"] = str(Path(shutil.which("cl.exe")).parent)

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

root = Path(__file__).resolve().parent
ext_modules = []
if os.environ.get("DANDAN_BUILD_CUDA", "1") != "0" and torch.version.cuda is not None and CUDA_HOME is not None:
    if os.environ.get("DANDAN_CUDA_DRIVER_LIB"):
        driver_candidates = [Path(os.environ["DANDAN_CUDA_DRIVER_LIB"])]
    elif os.name == "nt":
        installed = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA GPU Computing Toolkit/CUDA"
        driver_candidates = [Path(CUDA_HOME) / "lib/x64"] + sorted(installed.glob("v*/lib/x64"), reverse=True)
    else:
        driver_candidates = [Path(CUDA_HOME) / "lib64/stubs"]
    driver_dirs = [str(p) for p in driver_candidates if (p / ("cuda.lib" if os.name == "nt" else "libcuda.so")).is_file()]
    ext_modules.append(CUDAExtension(
        name="dandan_ops._C",
        sources=[str(Path("dandan_ops/csrc") / name) for name in ("operators.cu", "hgemm.cu", "hgemm_tma.cu", "flash_inference.cu", "elementwise.cu")],
        libraries=["cuda"],
        include_dirs=[str(root)],
        library_dirs=driver_dirs[:1],
        extra_compile_args={
            "cxx": ["/O2", "/std:c++17", "/utf-8"] if os.name == "nt" else ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "-lineinfo"] + (["-Xcompiler", "/utf-8"] if os.name == "nt" else []),
        },
    ))

setup(
    name="dandan-ops",
    version="0.1.0",
    description="Inference CUDA operators: FP16 GEMM, W4A16 INT4 and GQA attention for PyTorch",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md"],
    url="https://github.com/dandan20031031/dandan-ops",
    project_urls={"Issues": "https://github.com/dandan20031031/dandan-ops/issues"},
    packages=find_packages(include=["dandan_ops", "dandan_ops.*"]),
    package_data={"dandan_ops": ["profiles/*.json"]},
    python_requires=">=3.10",
    install_requires=["torch>=2.9"],
    extras_require={"myvllm": ["numpy>=1.26"], "dev": ["pytest>=8", "numpy>=1.26"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
