import os
import subprocess
import sys
import sysconfig
import copy
import time

from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Type

import setuptools


from torch.utils.cpp_extension import BuildExtension, CppExtension

from .utils import (
    debug_build_enabled,
    get_cmake_bin,
    found_ninja,
    get_max_jobs_for_parallel_build,
    get_rocm_cmake_args,
    get_all_files_in_dir
)


class CMakeExtension(setuptools.Extension):
    """
    CMake extension for building C++ extensions.
    """

    def __init__(
        self,
        name: str,
        cmake_path: Path,
        cmake_flags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name, sources=[])
        self.cmake_path: Path = cmake_path
        self.cmake_flags: List[str] = [
        ] if cmake_flags is None else cmake_flags

    def _build_cmake(self, build_dir: Path, install_dir: Path) -> None:
        # Make sure paths are str
        cmake_bin = str(get_cmake_bin())
        cmake_path = str(self.cmake_path)
        build_dir = str(build_dir)
        install_dir = str(install_dir)

        # CMake configure command
        build_type = "Debug" if debug_build_enabled() else "Release"
        configure_command = [
            cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        configure_command += get_rocm_cmake_args()
        configure_command += self.cmake_flags

        import pybind11

        pybind11_dir = Path(pybind11.__file__).resolve().parent
        pybind11_dir = pybind11_dir / "share" / "cmake" / "pybind11"
        configure_command.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake build and install commands
        build_command = [cmake_bin, "--build", build_dir, "--verbose"]
        install_command = [cmake_bin, "--install", build_dir, "--verbose"]

        # Check whether parallel build is restricted
        max_jobs = get_max_jobs_for_parallel_build()
        if found_ninja():
            configure_command.append("-GNinja")
        build_command.append("--parallel")
        if max_jobs > 0:
            build_command.append(str(max_jobs))

        # Run CMake commands
        start_time = time.perf_counter()
        for command in [configure_command, build_command, install_command]:
            print(f"Running command {' '.join(command)}")
            try:
                subprocess.run(command, cwd=build_dir, check=True)
            except (CalledProcessError, OSError) as e:
                raise RuntimeError(f"Error when running CMake: {e}")

        total_time = time.perf_counter() - start_time
        print(f"Time for build_ext: {total_time:.2f} seconds")


class CMakeBuildExtension(BuildExtension):
    """
    Custom build extension for CMake projects.
    """

    def run(self) -> None:
        # Build CMake extensions
        # Build CMake extensions
        for ext in self.extensions:
            package_path = Path(self.get_ext_fullpath(ext.name))
            install_dir = package_path.resolve().parent
            if isinstance(ext, CMakeExtension):
                print(f"Building CMake extension {ext.name}")
                # Set up incremental builds for CMake extensions
                build_dir = os.getenv("FLASH_CK_CMAKE_BUILD_DIR")
                if build_dir:
                    build_dir = Path(build_dir).resolve()
                else:
                    root_dir = Path(__file__).resolve().parent.parent
                    build_dir = root_dir / "build" / "cmake"

                # Ensure the directory exists
                build_dir.mkdir(parents=True, exist_ok=True)

                ext._build_cmake(
                    build_dir=build_dir,
                    install_dir=install_dir,
                )

        # Build non-CMake extensions as usual
        all_extensions = self.extensions
        self.extensions = [
            ext for ext in self.extensions if not isinstance(ext, CMakeExtension)
        ]
        super().run()
        self.extensions = all_extensions


def setup_pytorch_extension(csrc_source_files,
                            csrc_header_files,
                            common_header_files) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    sources = get_all_files_in_dir(
        Path(csrc_source_files), name_extension="cc")

    # Header files
    include_dirs = [
        common_header_files,
        csrc_header_files,
    ]

    # Compiler flags
    cxx_flags = ["-O3", "-fvisibility=hidden"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # Construct PyTorch ROCm extension

    return CppExtension(
        name="flash_ck_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={"cxx": cxx_flags}
    )
