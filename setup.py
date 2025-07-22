"""Installation script for flash_ck pytorch extensions."""

import setuptools

from pathlib import Path

from build_tools.utils import (
    TimedBdist, get_rocm_version, is_framework_available, get_fc_version)
from build_tools.build_pt import (
    CMakeExtension, setup_pytorch_extension, CMakeBuildExtension)


current_file_path = Path(__file__).parent.resolve()


def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library"""
    return CMakeExtension(
        name="flash_ck_common",
        cmake_path=current_file_path,
        cmake_flags=[]
    )


if __name__ == "__main__":

    # check pytorch avaliable
    if not is_framework_available("pytorch"):
        raise RuntimeError("PyTorch is not available")

    # check rocm version
    try:
        version = get_rocm_version()
    except FileNotFoundError:
        print("Could not determine ROCm version")
    else:
        if version < (5, 0):
            raise RuntimeError("FlashCK requires ROCm 6.0 or newer")

    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

    # add extmodules
    ext_modules = [setup_common_extension()]
    ext_modules.append(
        setup_pytorch_extension(
            current_file_path / "flashck" / "wrapper" / "pytorch" / "csrc",
            current_file_path,
            current_file_path,
        )
    )

    # Configure package
    setuptools.setup(
        name="flash_ck",
        include_package_data=True,  # include all files in the package
        package_data={"": ["version.txt"]},
        packages=setuptools.find_packages(
            where=".", include=["flashck", "flashck.*"]),
        package_dir={"flash_ck": "flashck"},
        version=get_fc_version(),
        description="FlashCK: fast and memory-efficient ck kernel",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=["Programming Language :: Python :: 3"],
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension, "bdist_wheel": TimedBdist},
        install_requires=["torch>=2.1"]
    )
