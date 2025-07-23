"""
FlashCK functional operations.

This module provides functional interfaces for various operations that can be used
as drop-in replacements for PyTorch's functional operations when FlashCK is available.
"""

from .norm import layer_norm_fwd
from ..utils import is_available

__all__ = ["layer_norm_fwd", "is_available"]
