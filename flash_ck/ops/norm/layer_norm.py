"""
Functional layer normalization operations for FlashCK.

This module provides functional interfaces for layer normalization that can be used
as drop-in replacements for torch.nn.functional.layer_norm_fwd.
"""

import numbers
import torch
import torch.nn.functional as F
from typing import Iterable, Optional, Union, List

from ...utils import is_available, get_flashck_functions, FlashCKOperationType

# Get FlashCK functions if available
_flashck_functions = get_flashck_functions(
    FlashCKOperationType.LAYER_NORM.value)

__all__ = ["layer_norm_fwd"]


def layer_norm_fwd(input: torch.Tensor,
                   normalized_shape: Union[int, Iterable[int]],
                   weight: Optional[torch.Tensor] = None,
                   bias: Optional[torch.Tensor] = None,
                   eps: float = 1e-5) -> torch.Tensor:
    """
    Applies layer normalization over a mini-batch of inputs.

    This function is compatible with torch.nn.functional.layer_norm and can be used
    as a drop-in replacement when FlashCK is available.

    Args:
        input: Input tensor of shape [*, normalized_shape[0], normalized_shape[1], ...]
        normalized_shape: Input shape from an expected input of size
            [*, normalized_shape[0], normalized_shape[1], ...]
            If a single integer is used, it is treated as a singleton list.
        weight: Optional learnable per-element affine parameters of shape normalized_shape.
            If None, weight is set to 1.0.
        bias: Optional learnable per-element affine parameters of shape normalized_shape.
            If None, bias is set to 0.0.
        eps: A value added to the denominator for numerical stability. Default: 1e-5

    Returns:
        Normalized tensor of the same shape as input.

    Example:
        >>> import torch
        >>> from flash_ck.ops.norm import layer_norm_fwd
        >>> 
        >>> # Basic usage
        >>> x = torch.randn(20, 5, 10, 10)
        >>> normalized_shape = (5, 10, 10)
        >>> y = layer_norm_fwd(x, normalized_shape)
        >>> 
        >>> # With learnable parameters
        >>> weight = torch.randn(normalized_shape)
        >>> bias = torch.randn(normalized_shape)
        >>> y = layer_norm_fwd(x, normalized_shape, weight, bias)
    """
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError(
            f"Expected input to be a torch.Tensor, got {type(input)}")

    if input.numel() == 0:
        raise ValueError("Input tensor cannot be empty")

    # Handle normalized_shape
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    # Validate normalized_shape
    if len(normalized_shape) == 0:
        raise ValueError("normalized_shape cannot be empty")

    if any(dim <= 0 for dim in normalized_shape):
        raise ValueError("All dimensions in normalized_shape must be positive")

    # Check if input dimensions match normalized_shape
    if len(input.shape) < len(normalized_shape):
        raise ValueError(
            f"Input tensor has {len(input.shape)} dimensions, but normalized_shape requires at least {len(normalized_shape)}")

    # Verify that the last dimensions match normalized_shape
    input_norm_dims = input.shape[-len(normalized_shape):]
    if input_norm_dims != normalized_shape:
        raise ValueError(f"Input tensor's last {len(normalized_shape)} dimensions {input_norm_dims} "
                         f"do not match normalized_shape {normalized_shape}")

    # Create default weight and bias if not provided
    if weight is None:
        weight = torch.ones(
            normalized_shape, dtype=input.dtype, device=input.device)
    else:
        if not isinstance(weight, torch.Tensor):
            raise TypeError(
                f"Expected weight to be a torch.Tensor, got {type(weight)}")
        if weight.shape != normalized_shape:
            raise ValueError(
                f"Weight shape {weight.shape} does not match normalized_shape {normalized_shape}")
        weight = weight.to(device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(
            normalized_shape, dtype=input.dtype, device=input.device)
    else:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(
                f"Expected bias to be a torch.Tensor, got {type(bias)}")
        if bias.shape != normalized_shape:
            raise ValueError(
                f"Bias shape {bias.shape} does not match normalized_shape {normalized_shape}")
        bias = bias.to(device=input.device, dtype=input.dtype)

    # Use FlashCK if available, otherwise fall back to PyTorch
    if is_available(FlashCKOperationType.LAYER_NORM.value) and _flashck_functions:
        try:
            return _flashck_functions['forward'](
                input,
                list(normalized_shape),
                weight,
                bias,
                eps,
            )
        except Exception as e:
            # If FlashCK fails, fall back to PyTorch with a warning
            import warnings
            warnings.warn(
                f"FlashCK layer_norm_fwd failed ({e}), falling back to PyTorch implementation")
            return F.layer_norm(input, normalized_shape, weight, bias, eps)
    else:
        return F.layer_norm(input, normalized_shape, weight, bias, eps)
