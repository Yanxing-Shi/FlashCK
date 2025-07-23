"""
FlashCK PyTorch Wrapper.

This package provides PyTorch-compatible interfaces for FlashCK operations,
including both functional and module-based APIs.
"""

# Import functional operations
from . import ops
from .ops import layer_norm_fwd

# Import neural network modules
from . import nn
from .nn import LayerNorm

# Import utilities
from .utils import is_available, get_available_operations, refresh_flashck_registry

# Refresh the FlashCK registry
refresh_flashck_registry()

__version__ = "0.1.0"

__all__ = [
    # Functional API
    "ops",
    "layer_norm_fwd",

    # Neural network modules
    # "nn",
    # "LayerNorm",

    # Utilities
    "is_available",
    "get_available_operations",
]
