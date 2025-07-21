"""Installation script for flash_ck pytorch extensions."""

#!/usr/bin/env python3

import os
from pathlib import Path

import setuptools


from torch.utils.cpp_extension import (CUDAExtension, BuildExtension)

from build_tools.utils import (
    get_all_files_in_dir,
    get_fc_version,
    get_rocm_version,
    get_rocm_archs,
    found_rocm,
    is_framework_available,
    ValidateSupportArchs,
    rename_cc_to_cu,
    get_all_files_in_dir

)


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / \
                (1024 ** 3)  # free memory in GB
            # each JOB peak memory cost is ~8-9GB when threads = 4
            max_num_jobs_memory = int(free_memory_gb / 9)

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
main_path = os.path.dirname(os.path.abspath(__file__))

# validate rocm and torch availability
if not is_framework_available("pytorch") or not found_rocm():
    raise RuntimeError(
        "FlashCK requires PyTorch and ROCm to be installed. "
        "Please install them before proceeding."
    )

# Validate ROCm architecture support for FlashCK
archs = os.getenv("GPU_ARCHS", "native").split(";")
ValidateSupportArchs(archs)

if archs != ['native']:
    cc_flags = [f"--offload-arch={arch}" for arch in archs]
else:
    arch = get_rocm_archs()
    cc_flags = [f"--offload-arch={arch}"]

# Source files
sources = get_all_files_in_dir(
    Path(main_path) / "flashck", name_extension="cc")

# rename .cc files to .cu
rename_sources = rename_cc_to_cu(sources)

# Header files
include_dirs = get_all_files_in_dir(
    Path(main_path) / "flashck", name_extension="h")
include_dirs.extend(
    [
        Path(main_path) / "3rdparty" / "composable_kernel" / "include",
        Path(main_path) / "3rdparty" /
        "composable_kernel" / "library" / "include",
        include_dirs,
    ]
)

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
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": cc_flags,
}

# Configure package
setuptools.setup(
    name="flash_ck",
    include_package_data=True,  # include all files in the package
    version=get_fc_version(),
    description="FlashCK: fast and memory-efficient ck kernel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=["Programming Language :: Python :: 3"],
    ext_modules=CUDAExtension(
        name="flash_ck",
        sources="",
        include_dirs=[str(d) for d in include_dirs],
        extra_compile_args=extra_compile_args,
    ),
    cmdclass={"build_ext": NinjaBuildExtension},
    install_requires=["torch>=2.1", "psutil"],
    tests_require=["pytest>=8.2.1"]
)
