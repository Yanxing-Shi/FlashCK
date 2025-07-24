"""Layer Normalization API."""

import numbers
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from typing import Iterable, Optional, Union

from ...utils import is_available, get_flashck_functions, FlashCKOperationType

# Get FlashCK functions if available
_flashck_functions = get_flashck_functions(
    FlashCKOperationType.LAYER_NORM.value)

__all__ = ["LayerNorm"]


class LayerNorm(torch.nn.Module):
    """
    Layer normalization module compatible with PyTorch's LayerNorm.

    This module applies layer normalization over a mini-batch of inputs as described in
    the paper "Layer Normalization" (https://arxiv.org/abs/1607.06450).

    Args:
        normalized_shape: Input shape from an expected input of size
            [* x normalized_shape[0] x normalized_shape[1] x ... x normalized_shape[-1]]
            If a single integer is used, it is treated as a singleton list.
        eps: A value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: A boolean value that when set to True, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: True
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        device: Device to place the module on
        dtype: Data type of the module parameters

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of additional dimensions
        - Output: :math:`(N, *)` (same shape as input)

    Examples:
        >>> import torch
        >>> from flashck.nn.norm import LayerNorm
        >>> 
        >>> # Basic usage with a single dimension
        >>> layer_norm = LayerNorm(768)
        >>> input = torch.randn(32, 128, 768)
        >>> output = layer_norm(input)
        >>> print(output.shape)  # torch.Size([32, 128, 768])
        >>> 
        >>> # Multi-dimensional normalization
        >>> layer_norm = LayerNorm((512, 256))
        >>> input = torch.randn(16, 512, 256)
        >>> output = layer_norm(input)
        >>> print(output.shape)  # torch.Size([16, 512, 256])
        >>> 
        >>> # Without learnable parameters
        >>> layer_norm = LayerNorm(768, elementwise_affine=False)
        >>> input = torch.randn(32, 128, 768)
        >>> output = layer_norm(input)
        >>> 
        >>> # Custom epsilon for numerical stability
        >>> layer_norm = LayerNorm(768, eps=1e-6)
        >>> input = torch.randn(32, 128, 768)
        >>> output = layer_norm(input)
        >>> 
        >>> # Without bias
        >>> layer_norm = LayerNorm(768, bias=False)
        >>> input = torch.randn(32, 128, 768)
        >>> output = layer_norm(input)
        >>> 
        >>> # Dynamic shape profiling (FlashCK specific)
        >>> layer_norm = LayerNorm(768)
        >>> input = torch.randn(32, 128, 768)
        >>> m_range = [64, 256]  # Support sequences from 64 to 256
        >>> output = layer_norm.forward_dynamic(input, m_range)
        >>> 
        >>> # Batch processing with different sequence lengths
        >>> layer_norm = LayerNorm(768)
        >>> inputs = [
        ...     torch.randn(32, 64, 768),   # Short sequences
        ...     torch.randn(32, 128, 768),  # Medium sequences
        ...     torch.randn(32, 256, 768),  # Long sequences
        ... ]
        >>> m_range = [64, 256]
        >>> outputs = [layer_norm.forward_dynamic(inp, m_range) for inp in inputs]
        >>> 
        >>> # Comparison with PyTorch's LayerNorm
        >>> import torch.nn as nn
        >>> torch_layer_norm = nn.LayerNorm(768)
        >>> flashck_layer_norm = LayerNorm(768)
        >>> 
        >>> # Copy parameters for fair comparison
        >>> flashck_layer_norm.weight.data = torch_layer_norm.weight.data.clone()
        >>> flashck_layer_norm.bias.data = torch_layer_norm.bias.data.clone()
        >>> 
        >>> input = torch.randn(32, 128, 768)
        >>> torch_output = torch_layer_norm(input)
        >>> flashck_output = flashck_layer_norm(input)
        >>> print(torch.allclose(torch_output, flashck_output, atol=1e-5))
        >>> 
        >>> # Use in a transformer-like model
        >>> class TransformerLayer(nn.Module):
        ...     def __init__(self, d_model=768):
        ...         super().__init__()
        ...         self.norm1 = LayerNorm(d_model)
        ...         self.norm2 = LayerNorm(d_model)
        ...         self.linear = nn.Linear(d_model, d_model)
        ...     
        ...     def forward(self, x):
        ...         # Pre-norm style
        ...         x = x + self.linear(self.norm1(x))
        ...         x = x + self.linear(self.norm2(x))
        ...         return x
        >>> 
        >>> model = TransformerLayer()
        >>> input = torch.randn(32, 128, 768)
        >>> output = model(input)

    Note:
        - This implementation is compatible with PyTorch's nn.LayerNorm
        - When FlashCK is available, it will use optimized kernels for better performance
        - Automatic fallback to PyTorch implementation if FlashCK is not available
        - The forward_dynamic method enables dynamic shape profiling for variable sequence lengths
        - Memory layout should be contiguous for optimal performance
    """

    def __init__(
        self,
        normalized_shape: Union[int, Iterable[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters to default values."""
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of layer normalization.

        Args:
            input: Input tensor of shape [..., normalized_shape]

        Returns:
            Output tensor of the same shape as input

        Examples:
            >>> layer_norm = LayerNorm(768)
            >>> input = torch.randn(32, 128, 768)
            >>> output = layer_norm(input)
            >>> print(output.shape)  # torch.Size([32, 128, 768])
            >>> 
            >>> # Verify normalization properties
            >>> normalized = layer_norm(input)
            >>> mean = normalized.mean(dim=-1, keepdim=True)
            >>> var = normalized.var(dim=-1, keepdim=True, unbiased=False)
            >>> print(f"Mean: {mean.abs().max():.6f}")  # Should be close to 0
            >>> print(f"Var: {var.mean():.6f}")         # Should be close to 1
        """
        if not self.elementwise_affine:
            # If no learnable parameters, use PyTorch's implementation
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, None, None, self.eps
            )

        # Ensure weight and bias are not None for FlashCK implementation
        weight = self.weight if self.weight is not None else torch.ones_like(
            input[..., :self.normalized_shape[0]])
        bias = self.bias if self.bias is not None else torch.zeros_like(
            input[..., :self.normalized_shape[0]])

        # Use FlashCK if available, otherwise fall back to PyTorch
        if is_available(FlashCKOperationType.LAYER_NORM.value) and _flashck_functions:
            try:
                return _flashck_functions['forward'](
                    input,
                    list(self.normalized_shape),
                    weight,
                    bias,
                    self.eps,
                )
            except Exception as e:
                # If FlashCK fails, fall back to PyTorch with a warning
                import warnings
                warnings.warn(
                    f"FlashCK layer_norm failed ({e}), falling back to PyTorch implementation")
                return torch.nn.functional.layer_norm(
                    input, self.normalized_shape, weight, bias, self.eps
                )
        else:
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    def forward_dynamic(self, input: torch.Tensor, m_range: list) -> torch.Tensor:
        """
        Forward pass of layer normalization with dynamic shape profiling.

        This method is specifically designed for scenarios where the input sequence length
        varies dynamically and you want to leverage FlashCK's dynamic profiling capabilities
        for optimal performance across different sequence lengths.

        Args:
            input: Input tensor of shape [..., normalized_shape]
            m_range: Range of sequence lengths for dynamic profiling [min_m, max_m]

        Returns:
            Output tensor of the same shape as input

        Examples:
            >>> layer_norm = LayerNorm(768)
            >>> 
            >>> # Single dynamic call
            >>> input = torch.randn(32, 128, 768)
            >>> m_range = [64, 256]  # Support sequences from 64 to 256
            >>> output = layer_norm.forward_dynamic(input, m_range)
            >>> print(output.shape)  # torch.Size([32, 128, 768])
            >>> 
            >>> # Batch processing with different sequence lengths
            >>> m_range = [64, 512]
            >>> test_lengths = [64, 128, 256, 512]
            >>> 
            >>> for seq_len in test_lengths:
            ...     input = torch.randn(16, seq_len, 768)
            ...     output = layer_norm.forward_dynamic(input, m_range)
            ...     print(f"seq_len={seq_len}: {output.shape}")
            >>> 
            >>> # Performance comparison
            >>> import time
            >>> input = torch.randn(32, 128, 768)
            >>> m_range = [64, 256]
            >>> 
            >>> # Warm up
            >>> for _ in range(10):
            ...     _ = layer_norm.forward_dynamic(input, m_range)
            >>> 
            >>> # Benchmark
            >>> start = time.time()
            >>> for _ in range(100):
            ...     _ = layer_norm.forward_dynamic(input, m_range)
            >>> dynamic_time = time.time() - start
            >>> 
            >>> start = time.time()
            >>> for _ in range(100):
            ...     _ = layer_norm(input)
            >>> static_time = time.time() - start
            >>> 
            >>> print(f"Dynamic time: {dynamic_time:.4f}s")
            >>> print(f"Static time: {static_time:.4f}s")
            >>> 

        Note:
            - The m_range parameter helps FlashCK optimize kernel selection
            - Use this method when dealing with variable sequence lengths
            - Falls back to regular forward() if FlashCK is not available
            - The profiling overhead is amortized over multiple calls
        """
        if not self.elementwise_affine:
            # If no learnable parameters, use PyTorch's implementation
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, None, None, self.eps
            )

        # Ensure weight and bias are not None for FlashCK implementation
        weight = self.weight if self.weight is not None else torch.ones_like(
            input[..., :self.normalized_shape[0]])
        bias = self.bias if self.bias is not None else torch.zeros_like(
            input[..., :self.normalized_shape[0]])

        # Use FlashCK dynamic implementation if available
        if is_available(FlashCKOperationType.LAYER_NORM.value) and _flashck_functions:
            try:
                return _flashck_functions['forward_dynamic'](
                    input,
                    list(self.normalized_shape),
                    weight,
                    bias,
                    m_range,
                    self.eps,
                )
            except Exception as e:
                # If FlashCK fails, fall back to regular forward
                import warnings
                warnings.warn(
                    f"FlashCK layer_norm_dynamic failed ({e}), falling back to regular forward")
                return self.forward(input)
        else:
            # Fall back to regular forward if FlashCK not available
            return self.forward(input)

    def extra_repr(self) -> str:
        """String representation of the module."""
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)
