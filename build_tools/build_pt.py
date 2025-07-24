import os
import subprocess
import sys
import sysconfig
import time

from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Type

import setuptools


from torch.utils.cpp_extension import BuildExtension

from .utils import (
    debug_build_enabled,
    get_cmake_bin,
    found_ninja,
    get_max_jobs_for_parallel_build,
    get_rocm_cmake_args,
    get_all_files_in_dir,
    get_torch_path
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
        cmake_prefix_path = f"{get_torch_path()}:/opt/rocm"
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
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}"
        ]
        # Remove -DCMAKE_PREFIX_PATH from get_rocm_cmake_args()
        rocm_args = [arg for arg in get_rocm_cmake_args(
        ) if not arg.startswith("-DCMAKE_PREFIX_PATH")]
        torch_dir = os.path.join(get_torch_path(), "Torch")
        configure_command.append(f"-DTorch_DIR={torch_dir}")
        configure_command += rocm_args
        configure_command += self.cmake_flags

        pybind11_dir = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()
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
                

                target_dir = install_dir / "flash_ck"
                target_dir.mkdir(exist_ok=True, parents=True)

                lib_dir = build_dir / "lib"
                if lib_dir.exists():
                    for so_file in lib_dir.glob("*.so"):
                        dst = target_dir / so_file.name
                        print(f"Copying {so_file} -> {dst}")
                        import shutil
                        shutil.copy2(so_file, dst)

        # Build non-CMake extensions as usual
        all_extensions = self.extensions
        self.extensions = [
            ext for ext in self.extensions if not isinstance(ext, CMakeExtension)
        ]
        super().run()
        self.extensions = all_extensions
