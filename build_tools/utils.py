#!/usr/bin/env python3

import functools
import importlib.metadata
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Tuple, Union


@functools.lru_cache(maxsize=None)
def get_max_jobs_for_parallel_build() -> int:
    """
    Get the optimal number of parallel jobs for building.

    Priority order:
    1. MAX_JOBS environment variable
    2. CPU count with safety margin

    Returns:
        Number of parallel jobs to use (0 means unlimited)
    """
    # Check environment variable first
    if env_jobs := os.getenv("MAX_JOBS"):
        try:
            return max(1, int(env_jobs))
        except ValueError:
            print(
                f"Warning: Invalid MAX_JOBS value: {env_jobs}")

    # Default: Use CPU count with safety margin (leave 1-2 cores free)
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count - 1) if cpu_count > 2 else 1


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
def get_rocm_archs() -> str:
    """
    Get supported ROCm GPU architectures.

    Returns:
        Semicolon-separated string of GPU architectures
    """
    # Check environment variable first
    if env_archs := os.getenv("FLASH_CK_ROCM_ARCHS"):
        return env_archs

    # Get ROCm version to determine default architectures
    version = get_rocm_version()

    # Default architectures based on ROCm version
    if version >= (6, 0):
        # ROCm 6.0+ supports latest architectures
        archs = "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101"
    elif version >= (5, 0):
        # ROCm 5.0+ supports most common architectures
        archs = "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
    else:
        # Older ROCm versions - basic support
        archs = "gfx900;gfx906;gfx908"

    # Cache in environment for future calls
    os.environ["FLASH_CK_ROCM_ARCHS"] = archs
    return archs


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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=root_path,
            check=True,
            text=True,
            timeout=5
        )
        commit = result.stdout.strip()
        if commit:
            version += f"+{commit}"
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        pass

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


def rename_cc_to_cu(cpp_files: List[Union[str, Path]]) -> List[str]:
    """
    Rename .cc files to .cu for CUDA compatibility.

    Args:
        cpp_files: List of .cc file paths to rename
    """
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")

    return [os.path.abspath(os.path.splitext(entry)[0] + ".cu") for entry in cpp_files]


def get_all_files_in_dir(path: Union[str, Path], name_extension: Optional[str] = None) -> List[Path]:
    """
    Get all files in a directory with optional file extension filtering.

    Args:
        path: Directory path to search for files
        name_extension: Optional file extension to filter by (e.g., "cpp")
    Returns:
        List of Path objects for all files found
    """
    path = Path(path)
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            if name_extension is not None and not name.endswith(f".{name_extension}"):
                continue
            all_files.append(Path(dirname, name))
    return all_files
