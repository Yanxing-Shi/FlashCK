"""
FlashCK normalization operations.

This module provides functional interfaces for normalization operations that can be used
as drop-in replacements for PyTorch's normalization functions when FlashCK is available.
"""

from .layer_norm import layer_norm_fwd
from ...utils.utils import is_available

__all__ = ["layer_norm_fwd", "is_available"]
