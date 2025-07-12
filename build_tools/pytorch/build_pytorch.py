"""PyTorch related extensions."""

#!/usr/bin/env python3

import os
from pathlib import Path


import setuptools

from .utils import all_files_in_dir, cuda_version, get_cuda_include_dirs, debug_build_enabled
from typing import List

from torch.utils.cpp_extension import CUDAExtension


def rename_cc_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    sources = all_files_in_dir(Path(csrc_source_files), name_extension="cc")

    # Header files
    include_dirs = get_hip_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
        ]
    )

    # rename .cc files to .cu
    rename_cc_to_cu(sources)

    # Compiler flags
    cc_flags += ["-O3", "-std=c++17",
                 "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                 "-fgpu-flush-denormals-to-zero",
                 "-DCK_ENABLE_BF16",
                 "-DCK_ENABLE_BF8",
                 "-DCK_ENABLE_FP16",
                 "-DCK_ENABLE_FP32",
                 "-DCK_ENABLE_FP64",
                 "-DCK_ENABLE_FP8",
                 "-DCK_ENABLE_INT8",
                 "-DCK_USE_XDL",
                 "-DUSE_PROF_API=1",
                 # "-DFLASHATTENTION_DISABLE_BACKWARD",
                 "-D__HIP_PLATFORM_HCC__=1"]

    hip_version = get_rocm_version()
    if hip_version < (6, 0):
        raise RuntimeError(
            "FlashCK requires ROCm 6.0 or newer")
    if hip_version > (5, 5, 0):
        cc_flags += ["-mllvm", "--amdgpu-early-inline-all=1"]
    if hip_version > (5, 7, 23302):
        cc_flags += ["-fno-offload-uniform-block"]
    if hip_version > (6, 1, 40090):
        cc_flags += ["-mllvm", "-enable-post-misched=0"]
    if hip_version > (6, 2, 41132):
        cc_flags += ["-mllvm", "-amdgpu-early-inline-all=true",
                     "-mllvm", "-amdgpu-function-calls=false"]
    if hip_version > (6, 2, 41133) and hip_version < (6, 3, 0):
        cc_flags += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"] + generator_flag,
        "nvcc": cc_flags + generator_flag,
    }

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]

    return CUDAExtension(
        name="flash_ck_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={"cxx": cxx_flags},
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
