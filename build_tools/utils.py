"""Installation script."""

import os
import shutil
from pathlib import Path
from typing import Tuple
import functools


@functools.lru_cache(maxsize=None)
def debug_build_enabled() -> bool:
    """Whether to build with a debug configuration"""
    for arg in sys.argv:
        if arg == "--debug":
            sys.argv.remove(arg)
            return True
    if int(os.getenv("FC_BUILD_DEBUG", "0")):
        return True
    return False

@functools.lru_cache(maxsize=None)
def get_max_jobs_for_parallel_build() -> int:
    """Number of parallel jobs for Nina build"""

    # Default: maximum parallel jobs
    num_jobs = 0

    # Check environment variable
    if os.getenv("FC_BUILD_MAX_JOBS"):
        num_jobs = int(os.getenv("FC_BUILD_MAX_JOBS"))
    elif os.getenv("MAX_JOBS"):
        num_jobs = int(os.getenv("MAX_JOBS"))

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--parallel="):
            num_jobs = int(arg.replace("--parallel=", ""))
            sys.argv.remove(arg)

    return num_jobs









def found_pybind11(timeout: Optional[float] = 5.0) -> bool:
    """Check if pybind11 is available through Python package or CMake.
    
    Args:
        timeout: Maximum execution time for CMake check (seconds)
        
    Returns:
        bool: True if pybind11 is detectable through any method
    """
    
    # 1. Python package check with version validation
    with suppress(ImportError):
        import pybind11
        logger.debug("Found pybind11 Python package (v%s)", pybind11.__version__)
        return True
    
    # 2. CMake-based detection with enhanced error handling
    if not found_cmake():
        logger.warning("CMake not available for pybind11 system detection")
        return False

    cmake_cmd = [
        "cmake",
        "-P",  # Use script mode for better cross-version compatibility
        f"{Path(__file__).parent}/cmake/FindPybind11.cmake"
    ]
    
    try:
        result = subprocess.run(
            cmake_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
            text=True
        )
        if "pybind11_FOUND:TRUE" in result.stdout:
            logger.debug("Found pybind11 via CMake: %s", result.stdout)
            return True
    except (CalledProcessError, TimeoutExpired, OSError) as e:
        logger.error("CMake detection failed: %s", getattr(e, 'output', str(e)))
    
    logger.info("Pybind11 not found through any detection method")
    return False


@functools.lru_cache(maxsize=None)
def rocm_path() -> Tuple[str, str]:
    """Get ROCm root path and HIPCC binary path as a tuple.
    
    Raises:
        FileNotFoundError: If HIPCC compiler cannot be located.
    """
    # Environment variable priority: HIP_PATH -> ROCM_PATH -> system PATH -> default paths
    rocm_home = None
    hipcc_bin: Optional[Path] = None

    # 1. Check HIP_PATH environment variable
    if os.getenv("HIP_PATH"):
        hip_path = Path(os.getenv("HIP_PATH"))
        hipcc_bin = hip_path / "bin" / "hipcc"
        if hipcc_bin.is_file():
            rocm_home = hip_path.parent  # HIP_PATH typically points to rocm/hip directory
        else:
            hipcc_bin = None

    # 2. Check ROCM_PATH environment variable
    if not hipcc_bin and os.getenv("ROCM_PATH"):
        rocm_path = Path(os.getenv("ROCM_PATH"))
        hipcc_bin = rocm_path / "bin" / "hipcc"
        if hipcc_bin.is_file():
            rocm_home = rocm_path
        else:
            hipcc_bin = None

    # 3. Search system PATH for hipcc
    if not hipcc_bin:
        hipcc_bin = shutil.which("hipcc")
        if hipcc_bin:
            hipcc_bin = Path(hipcc_bin)
            # Infer ROCm root from path (e.g., /opt/rocm-6.2.0/bin/hipcc -> /opt/rocm-6.2.0)
            rocm_home = hipcc_bin.parent.parent.resolve()

    # 4. Check default installation paths
    if not hipcc_bin:
        default_paths = [
            Path("/opt/rocm"),          # Standard installation
            Path("/opt/rocm-6.2.0"),    # Versioned path (common for ROCm 6.2)
            Path("/opt/rocm-6.3.0")     # Latest version path
        ]
        for path in default_paths:
            candidate = path / "bin" / "hipcc"
            if candidate.is_file():
                rocm_home = path
                hipcc_bin = candidate
                break

    # Validation
    if not hipcc_bin or not hipcc_bin.is_file():
        raise FileNotFoundError(
            f"HIPCC not found. Checked locations:\n"
            f"- HIP_PATH: {os.getenv('HIP_PATH')}\n"
            f"- ROCM_PATH: {os.getenv('ROCM_PATH')}\n"
            f"- System PATH: {shutil.which('hipcc')}\n"
            f"- Default paths: {default_paths}"
        )

    return str(rocm_home), str(hipcc_bin)


def rocm_version() -> Optional[Tuple[int, ...]]:
    """Detect ROCm version as a semantic version tuple (major, minor, patch).
    
    Returns:
        Tuple[int, ...] or None: Version tuple if ROCm is detected, otherwise None.
    """
    try:
        # 1. Locate HIPCC compiler
        rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")  # Default installation path
        hipcc_bin = os.path.join(rocm_path, "bin", "hipcc")
        
        # Fallback search using 'which' command
        if not os.path.isfile(hipcc_bin):
            which_result = subprocess.run(
                ["which", "hipcc"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            if which_result.returncode == 0:
                hipcc_bin = which_result.stdout.strip()
            else:
                return None

        # 2. Query compiler version
        version_output = subprocess.run(
            [hipcc_bin, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True
        )
        
        # 3. Pattern matching for different version formats
        version_patterns = [
            r"HIP version:\s*([\d.]+)",       # Standard format (ROCm 5.x+)
            r"clang version.*HIP\s*([\d.]+)"  # Legacy format (pre-ROCm 5.x)
        ]
        
        version_match = None
        for pattern in version_patterns:
            version_match = re.search(pattern, version_output.stdout)
            if version_match:
                break
                
        if not version_match:
            return None
            
        # 4. Parse version components
        version_components = list(map(int, version_match.group(1).split(".")))
        if len(version_components) < 2:
            return None  # Require at least major.minor
        
        # Return (major, minor, patch) if available
        return tuple(version_components[:3])  
        
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
        # Handle execution errors, missing files, or permission issues
        return None


def get_frameworks() -> List[str]:
    """DL frameworks to build support for"""
    _frameworks: List[str] = []
    supported_frameworks = ["pytorch"]

    # Check environment variable
    if os.getenv("NVTE_FRAMEWORK"):
        _frameworks.extend(os.getenv("NVTE_FRAMEWORK").split(","))

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--framework="):
            _frameworks.extend(arg.replace("--framework=", "").split(","))
            sys.argv.remove(arg)

    # Detect installed frameworks if not explicitly specified
    if not _frameworks:
        try:
            import torch
        except ImportError:
            pass
        else:
            _frameworks.append("pytorch")

    # Special framework names
    if "all" in _frameworks:
        _frameworks = supported_frameworks.copy()
    if "none" in _frameworks:
        _frameworks = []

    # Check that frameworks are valid
    _frameworks = [framework.lower() for framework in _frameworks]
    for framework in _frameworks:
        if framework not in supported_frameworks:
            raise ValueError(f"Transformer Engine does not support framework={framework}")

    return _frameworks

def install_and_import(package):
    """Install a package via pip (if not already installed) and import into globals."""
    main_package = package.split("[")[0]
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    globals()[main_package] = importlib.import_module(main_package)
    
def uninstall_te_wheel_packages():
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "flashck_torch
        ]
    )