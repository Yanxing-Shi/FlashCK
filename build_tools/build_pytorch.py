"""PyTorch related extensions."""

import os
from pathlib import Path

import setuptools

from torch.utils.cpp_extension import CUDAExtension

def setup_pytorch_extension(
    csrc_tpl_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""
    
    
    
    # Version-dependent CUDA options
    try:
        version = hip_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version < (6, 0):
            raise RuntimeError("FlashCK requires ROCm 6.0 or newer")
        nvcc_flags.extend(
            (
                "--threads",
                os.getenv("NVTE_BUILD_THREADS_PER_JOB", "1"),
            )
        )
        
    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    return CUDAExtension(
        name="flashck_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )