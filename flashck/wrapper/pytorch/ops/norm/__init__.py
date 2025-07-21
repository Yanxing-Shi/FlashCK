"""
FlashCK normalization operations.

This module provides functional interfaces for normalization operations that can be used
as drop-in replacements for PyTorch's normalization functions when FlashCK is available.
"""

from .layer_norm import layer_norm, layer_norm_dynamic
from ...utils import is_flashck_available

__all__ = ["layer_norm", "layer_norm_dynamic", "is_flashck_available"]
