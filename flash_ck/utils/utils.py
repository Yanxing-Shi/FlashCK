"""
Utility functions for FlashCK PyTorch wrapper.

This module provides utility functions that are shared across different operation modules
and can be easily extended to support new operations like FMHA, GEMM, etc.
"""

import warnings
from typing import Dict, Optional, Any, Callable
from enum import Enum

__all__ = [
    "FlashCKOperationType",
    "is_available",
    "get_flashck_functions",
    "register_flashck_functions",
    "get_available_operations",
    "get_operation_function",
    "with_flashck_fallback"
]

class FlashCKOperationType(Enum):
    """Enumeration of supported FlashCK operation types."""
    LAYER_NORM = "layer_norm"
    FMHA = "fmha"
    GEMM = "gemm"


# Global registry for FlashCK functions
_FLASHCK_FUNCTIONS: Dict[str, Dict[str, Callable]] = {}
_FLASHCK_AVAILABILITY: Dict[str, bool] = {}


def _try_import_flashck_functions():
    """Try to import FlashCK functions and register them."""
    global _FLASHCK_FUNCTIONS, _FLASHCK_AVAILABILITY

    # Try to import layer normalization functions
    try:
        from flash_ck import layer_norm_fwd
        _FLASHCK_FUNCTIONS[FlashCKOperationType.LAYER_NORM.value] = {
            'forward': layer_norm_fwd
        }
        _FLASHCK_AVAILABILITY[FlashCKOperationType.LAYER_NORM.value] = True
    except ImportError:
        _FLASHCK_AVAILABILITY[FlashCKOperationType.LAYER_NORM.value] = False
        warnings.warn(
            "FlashCK layer normalization functions not available, falling back to PyTorch implementation")

    # Try to import FMHA functions (placeholder for future implementation)
    try:
        # from flash_ck_torch import fmha_fwd
        # _FLASHCK_FUNCTIONS[FlashCKOperationType.FMHA.value] = {
        #     'forward': fmha_fwd,
        # }
        # _FLASHCK_AVAILABILITY[FlashCKOperationType.FMHA.value] = True
        _FLASHCK_AVAILABILITY[FlashCKOperationType.FMHA.value] = False
    except ImportError:
        _FLASHCK_AVAILABILITY[FlashCKOperationType.FMHA.value] = False

    # Try to import GEMM functions (placeholder for future implementation)
    try:
        # from flash_ck_torch import gemm_fwd, gemm_bwd
        # _FLASHCK_FUNCTIONS[FlashCKOperationType.GEMM.value] = {
        #     'forward': gemm_fwd,
        #     'backward': gemm_bwd,
        # }
        # _FLASHCK_AVAILABILITY[FlashCKOperationType.GEMM.value] = True
        _FLASHCK_AVAILABILITY[FlashCKOperationType.GEMM.value] = False
    except ImportError:
        _FLASHCK_AVAILABILITY[FlashCKOperationType.GEMM.value] = False


# Initialize FlashCK functions on module import
_try_import_flashck_functions()


def is_available(operation_type: Optional[str] = None) -> bool:
    """
    Check if FlashCK is available for specific operation types.

    Args:
        operation_type: The operation type to check. If None, checks if any FlashCK operations are available.

    Returns:
        True if FlashCK is available for the specified operation, False otherwise.
    """
    if operation_type is None:
        return any(_FLASHCK_AVAILABILITY.GetAllValues()())

    return _FLASHCK_AVAILABILITY.get(operation_type, False)


def get_flashck_functions(operation_type: str) -> Optional[Dict[str, Callable]]:
    """
    Get FlashCK functions for a specific operation type.

    Args:
        operation_type: The operation type (e.g., 'layer_norm', 'fmha', 'gemm').

    Returns:
        A dictionary containing FlashCK functions if available, None otherwise.
    """
    if not is_available(operation_type):
        return None

    return _FLASHCK_FUNCTIONS.get(operation_type, None)


def register_flashck_functions(operation_type: str, functions: Dict[str, Callable]) -> None:
    """
    Register FlashCK functions for a specific operation type.

    This function allows for dynamic registration of new FlashCK operations.

    Args:
        operation_type: The operation type to register.
        functions: Dictionary mapping function names to callable functions.
    """
    _FLASHCK_FUNCTIONS[operation_type] = functions
    _FLASHCK_AVAILABILITY[operation_type] = True


def get_available_operations() -> Dict[str, bool]:
    """
    Get a dictionary of all available FlashCK operations.

    Returns:
        Dictionary mapping operation types to their availability status.
    """
    return _FLASHCK_AVAILABILITY.copy()


def get_operation_function(operation_type: str, function_name: str) -> Optional[Callable]:
    """
    Get a specific function for an operation type.

    Args:
        operation_type: The operation type (e.g., 'layer_norm', 'fmha').
        function_name: The function name (e.g., 'forward', 'backward').

    Returns:
        The function if available, None otherwise.
    """
    functions = get_flashck_functions(operation_type)
    if functions is None:
        return None

    return functions.get(function_name, None)


def with_flashck_fallback(operation_type: str, function_name: str, fallback_func: Callable):
    """
    Decorator to provide automatic fallback to PyTorch implementation.

    Args:
        operation_type: The FlashCK operation type.
        function_name: The FlashCK function name.
        fallback_func: The fallback function to use if FlashCK is not available.

    Returns:
        A decorator function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            flashck_func = get_operation_function(
                operation_type, function_name)
            if flashck_func is not None:
                try:
                    return flashck_func(*args, **kwargs)
                except Exception as e:
                    warnings.warn(
                        f"FlashCK {operation_type}.{function_name} failed ({e}), "
                        f"falling back to PyTorch implementation")
                    return fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator



