#!/usr/bin/env python3

import functools
import importlib.metadata
import multiprocessing
import os
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Tuple, Union

from wheel.bdist_wheel import bdist_wheel


@functools.lru_cache(maxsize=None)
def debug_build_enabled() -> bool:
    """Whether to build with a debug configuration"""
    return bool(int(os.getenv("FLASH_CK_BUILD_DEBUG", "0")))


@functools.lru_cache(maxsize=None)
def get_max_jobs_for_parallel_build() -> int:
    """
    Get the optimal number of parallel jobs for building.

    Priority order:
    1. FLASH_CK_BUILD_MAX_JOBS environment variable
    2. --parallel= command line argument
    3. CPU count with safety margin

    Returns:
        Number of parallel jobs to use (0 means unlimited)
    """
    # Check environment variable first
    if env_jobs := os.getenv("FLASH_CK_BUILD_MAX_JOBS"):
        try:
            return max(1, int(env_jobs))
        except ValueError:
            print(
                f"Warning: Invalid FLASH_CK_BUILD_MAX_JOBS value: {env_jobs}")

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--parallel="):
            try:
                jobs = int(arg.replace("--parallel=", ""))
                sys.argv.remove(arg)
                return max(1, jobs)
            except ValueError:
                print(f"Warning: Invalid --parallel value: {arg}")

    # Default: Use CPU count with safety margin (leave 1-2 cores free)
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count - 1) if cpu_count > 2 else 1


@functools.lru_cache(maxsize=None)
def get_cmake_bin() -> str:
    """
    Get the CMake binary path.

    Returns:
        Path to CMake binary

    Raises:
        FileNotFoundError: If CMake is not found
    """
    cmake_path = shutil.which("cmake")
    if cmake_path is None:
        raise FileNotFoundError("CMake not found in PATH")
    return cmake_path


@functools.lru_cache(maxsize=None)
def found_cmake() -> bool:
    """
    Check if valid CMake is available.

    CMake 3.18 or newer is required for ROCm support.

    Returns:
        True if valid CMake is available, False otherwise
    """
    try:
        cmake_path = get_cmake_bin()
    except FileNotFoundError:
        return False

    try:
        # Query CMake for version info
        result = subprocess.run(
            [cmake_path, "--version"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10
        )

        # Parse version
        match = re.search(r"version\s*([\d.]+)", result.stdout)
        if not match:
            return False

        version_str = match.group(1)
        version_parts = version_str.split(".")
        version = tuple(int(v) for v in version_parts[:2])  # Only major.minor

        return version >= (3, 18)

    except (subprocess.TimeoutExpired, CalledProcessError, ValueError):
        return False


@functools.lru_cache(maxsize=None)
def found_ninja() -> bool:
    """
    Check if Ninja build system is available.

    Returns:
        True if Ninja is available, False otherwise
    """
    return shutil.which("ninja") is not None


@functools.lru_cache(maxsize=None)
def found_pybind11() -> bool:
    """
    Check if pybind11 is available.

    Checks both Python package and CMake findability.

    Returns:
        True if pybind11 is available, False otherwise
    """
    # Check if Python package is installed
    try:
        import pybind11  # noqa: F401
        return True
    except ImportError:
        pass

    # Check if CMake can find pybind11
    if not found_cmake():
        return False

    try:
        subprocess.run(
            [
                get_cmake_bin(),
                "--find-package",
                "-DMODE=EXIST",
                "-DNAME=pybind11",
                "-DCOMPILER_ID=CXX",
                "-DLANGUAGE=CXX",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10
        )
        return True
    except (CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False


@functools.lru_cache(maxsize=None)
def get_rocm_path() -> Path:
    """
    Get the ROCm installation path.

    Returns:
        Path to ROCm installation

    Raises:
        FileNotFoundError: If ROCm is not found
    """
    # Try environment variable first
    if rocm_home := os.getenv("ROCM_PATH"):
        rocm_path = Path(rocm_home)
        if rocm_path.exists():
            return rocm_path

    # Try common installation paths
    common_paths = [
        Path("/opt/rocm"),
        Path("/usr/local/rocm"),
        Path("/usr/rocm"),
    ]

    for path in common_paths:
        if path.exists() and (path / "bin" / "hipcc").exists():
            return path

    raise FileNotFoundError("ROCm installation not found")


@functools.lru_cache(maxsize=None)
def get_hipcc_path() -> Path:
    """
    Get the HIP compiler (hipcc) path.

    Returns:
        Path to hipcc binary

    Raises:
        FileNotFoundError: If hipcc is not found
    """
    # Try finding hipcc in PATH first
    if hipcc_bin := shutil.which("hipcc"):
        return Path(hipcc_bin)

    # Try finding in ROCm installation
    try:
        rocm_root = get_rocm_path()
        hipcc_bin = rocm_root / "bin" / "hipcc"
        if hipcc_bin.exists():
            return hipcc_bin
    except FileNotFoundError:
        pass

    raise FileNotFoundError("HIP compiler (hipcc) not found")


@functools.lru_cache(maxsize=None)
def get_torch_path() -> str:
    """
    Get the CMake prefix path for PyTorch.

    Returns:
        Path string for CMAKE_PREFIX_PATH to find TorchConfig.cmake
    """
    import torch
    return torch.utils.cmake_prefix_path


@functools.lru_cache(maxsize=None)
def get_gpu_name_by_id(gpu_id: int = 0) -> str:
    """Retrieve GPU name (e.g. gfx90a) by device ID"""
    GPU_NAME_PATTERN = re.compile(r"Name:\s*(gfx\d+\w*)")
    try:
        output = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.PIPE, timeout=5
        )
        if matches := GPU_NAME_PATTERN.finditer(output):
            gpu_list = [m.group(1) for m in matches]
            return gpu_list[gpu_id] if gpu_id < len(gpu_list) else ""

        return ""

    except subprocess.CalledProcessError as e:
        print(f"GPU query failed (exit {e.returncode}): {e.stderr.strip()}")
    except FileNotFoundError:
        print("ROCm tools not installed (requires rocminfo)")
    except subprocess.TimeoutExpired:
        print("GPU query timeout (5s)")
    except Exception as e:
        print(f"GPU detection error: {str(e)}")

    return ""


@functools.lru_cache(maxsize=None)
def get_rocm_archs() -> str:
    """
    Get supported ROCm GPU architectures.

    Returns:
        Semicolon-separated string of GPU architectures
    """
    env_archs = os.getenv("FLASH_CK_ROCM_ARCHS", "native")

    if env_archs == "native":
        arch = get_gpu_name_by_id()
        return arch if arch else "native"
    else:
        # Accept comma/semicolon separated env, or fallback to default list
        if isinstance(env_archs, str):
            archs = [a.strip()
                     for a in re.split(r"[;,]", env_archs) if a.strip()]
        else:
            archs = ["gfx90a", "gfx950", "gfx942"]
        return ";".join(archs) if archs else "gfx90a"


@functools.lru_cache(maxsize=None)
def get_rocm_version() -> Tuple[int, ...]:
    """
    Get ROCm version as a tuple.

    Returns:
        ROCm version as (major, minor, patch) tuple

    Raises:
        RuntimeError: If ROCm version cannot be determined
    """
    # Try getting version from hipcc
    try:
        hipcc_bin = get_hipcc_path()
        result = subprocess.run(
            [str(hipcc_bin), "--version"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10
        )

        # Parse HIP version from output
        match = re.search(r"HIP version:\s*([\d.]+)", result.stdout)
        if match:
            version_str = match.group(1)
            version_parts = version_str.split(".")
            return tuple(int(v) for v in version_parts)

    except (FileNotFoundError, CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Try getting version from ROCm installation
    try:
        rocm_root = get_rocm_path()
        version_file = rocm_root / ".info" / "version"
        if version_file.exists():
            version_str = version_file.read_text().strip()
            version_parts = version_str.split(".")
            return tuple(int(v) for v in version_parts)

    except (FileNotFoundError, OSError):
        pass

    # Try getting version from ROCm packages
    try:
        # Try common ROCm packages
        packages = ["rocm-dev", "hip-dev", "rocm-runtime"]
        for package in packages:
            try:
                version_str = importlib.metadata.version(package)
                version_parts = version_str.split(".")
                return tuple(int(v) for v in version_parts if v.isdigit())
            except importlib.metadata.PackageNotFoundError:
                continue

    except Exception:
        pass

    raise RuntimeError("Could not determine ROCm version")


@functools.lru_cache(maxsize=None)
def found_rocm() -> bool:
    """
    Check if ROCm is available and properly configured.

    Returns:
        True if ROCm is available, False otherwise
    """
    try:
        # Check if ROCm path exists
        rocm_root = get_rocm_path()

        # Check if hipcc is available
        hipcc_bin = get_hipcc_path()

        # Verify hipcc works
        result = subprocess.run(
            [str(hipcc_bin), "--version"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10
        )

        return "HIP" in result.stdout

    except (FileNotFoundError, CalledProcessError, subprocess.TimeoutExpired):
        return False


@functools.lru_cache(maxsize=None)
def is_framework_available(framework: str) -> bool:
    """
    Check if a specific framework is available with ROCm support.

    Args:
        framework: Name of the framework to check

    Returns:
        True if framework is installed with ROCm support, False otherwise
    """
    framework = framework.lower().strip()

    # Ensure ROCm is available first
    if not found_rocm():
        return False

    if framework == "pytorch":
        try:
            import torch

            # Check if PyTorch has ROCm support
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                # Check if it's actually ROCm (not CUDA)
                device_name = torch.cuda.get_device_name(
                    0) if torch.cuda.device_count() > 0 else ""
                return "AMD" in device_name or "gfx" in device_name.lower()

            # Also check torch.version for ROCm
            if hasattr(torch, "version"):
                version_info = str(torch.version.cuda) if hasattr(
                    torch.version, "cuda") else ""
                return "rocm" in version_info.lower()

        except (ImportError, AttributeError):
            pass

    return False


@functools.lru_cache(maxsize=None)
def get_rocm_cmake_args() -> List[str]:
    """
    Get CMake arguments for ROCm configuration.

    Returns:
        List of CMake arguments for ROCm build
    """
    args = []

    try:
        # Add ROCm path
        rocm_root = get_rocm_path()
        args.extend([
            f"-DCMAKE_PREFIX_PATH={rocm_root}",
            f"-DROCM_PATH={rocm_root}",
            f"-DHIP_PATH={rocm_root}",
        ])

        # Add HIP compiler
        hipcc_bin = get_hipcc_path()
        args.extend([
            f"-DCMAKE_CXX_COMPILER={hipcc_bin}"])

        # Add GPU architectures
        archs = get_rocm_archs()
        args.extend([
            f"-DGPU_TARGETS={archs}"])

    except FileNotFoundError as e:
        raise RuntimeError(f"ROCm configuration failed: {e}")

    return args


@functools.lru_cache(maxsize=None)
def get_rocm_env_vars() -> dict:
    """
    Get environment variables for ROCm configuration.

    Returns:
        Dictionary of environment variables for ROCm
    """
    env_vars = {}

    try:
        # ROCm paths
        rocm_root = get_rocm_path()
        env_vars.update({
            "ROCM_PATH": str(rocm_root),
            "HIP_PATH": str(rocm_root),
            "ROCM_HOME": str(rocm_root),
        })

        # HIP compiler
        hipcc_bin = get_hipcc_path()
        env_vars.update({
            "HIP_COMPILER": str(hipcc_bin),
            "HIPCC": str(hipcc_bin),
        })

        # GPU architectures
        archs = get_rocm_archs()
        env_vars.update({
            "GPU_TARGETS": archs,
            "AMDGPU_TARGETS": archs,
            "FLASH_CK_ROCM_ARCHS": archs,
        })

        # ROCm version
        version = get_rocm_version()
        version_str = ".".join(str(v) for v in version)
        env_vars["ROCM_VERSION"] = version_str

    except FileNotFoundError as e:
        raise RuntimeError(f"ROCm environment configuration failed: {e}")

    return env_vars


@functools.lru_cache(maxsize=None)
def get_fc_version() -> str:
    """
    Get FlashCK version string with commit hash.

    Returns:
        Version string in format: X.Y.Z+commit_hash
    """
    root_path = Path(__file__).resolve().parent

    # Get base version
    try:
        with open(root_path / "version.txt", "r", encoding="utf-8") as f:
            version = f.read().strip()
    except (OSError, UnicodeDecodeError):
        return "unknown"

    # Add git commit hash
    # try:
    #     result = subprocess.run(
    #         ["git", "rev-parse", "--short", "HEAD"],
    #         capture_output=True,
    #         cwd=root_path,
    #         check=True,
    #         text=True,
    #         timeout=5
    #     )
    #     commit = result.stdout.strip()
    #     if commit:
    #         version += f"+{commit}"
    # except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
    #     pass

    return version


def ValidateSupportArchs(archs):
    """
    Validate and update GPU architectures for FlashCK.

    Args:
        archs: List of GPU architectures to validate

    Raises:
        AssertionError: If any architecture is invalid or not supported 
    """

    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx950", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by FlashCK. Allowed archs: {allowed_archs}"


def rename_cc_to_cu(cpp_files: List[Path]):
    """
    Rename .cc files to .cu for CUDA compatibility.

    Args:
        cpp_files: List of Path objects representing .cc files
    """
    for entry in cpp_files:
        new_name = entry.with_suffix(".cu")
        if new_name != entry:
            try:
                entry.rename(new_name)
                print(f"Renamed {entry} to {new_name}")
            except OSError as e:
                print(f"Error renaming {entry} to {new_name}: {e}")
        else:
            print(f"No renaming needed for {entry}")


class TimedBdist(bdist_wheel):
    """Helper class to measure build time"""

    def run(self):
        start_time = time.perf_counter()
        super().run()
        total_time = time.perf_counter() - start_time
        print(f"Total time for bdist_wheel: {total_time:.2f} seconds")


def get_all_files_in_dir(path, name_extension=None):
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            if name_extension is not None and not name.endswith(f".{name_extension}"):
                continue
            all_files.append(Path(dirname, name))
    return all_files


def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")
