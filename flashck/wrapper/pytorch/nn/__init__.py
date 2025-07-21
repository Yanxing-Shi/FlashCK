"""
FlashCK neural network modules.

This module provides neural network layers and modules that can be used as drop-in
replacements for PyTorch's neural network modules when FlashCK is available.
"""

from .norm import LayerNorm

__all__ = ["LayerNorm"]
