"""
FlashCK PyTorch Wrapper.

This package provides PyTorch-compatible interfaces for FlashCK operations,
including both functional and module-based APIs.
"""

# Import functional operations
from . import ops
from .ops import layer_norm, layer_norm_dynamic

# Import neural network modules
from . import nn
from .nn import LayerNorm

# Import utilities
from .utils import is_flashck_available, get_available_operations

__version__ = "0.1.0"

__all__ = [
    # Functional API
    "ops",
    "layer_norm",
    "layer_norm_dynamic",

    # Neural network modules
    "nn",
    "LayerNorm",

    # Utilities
    "is_flashck_available",
    "get_available_operations",
]
