"""
FlashCK functional operations.

This module provides functional interfaces for various operations that can be used
as drop-in replacements for PyTorch's functional operations when FlashCK is available.
"""

from .norm import layer_norm, layer_norm_dynamic
from ..utils import is_flashck_available

__all__ = ["layer_norm", "layer_norm_dynamic", "is_flashck_available"]
