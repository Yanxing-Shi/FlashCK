"""
FlashCK neural network normalization layers.

This module provides normalization layers that can be used as drop-in replacements
for PyTorch's normalization layers when FlashCK is available.
"""

from .layer_norm import LayerNorm

__all__ = ["LayerNorm"]
