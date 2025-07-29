""" test FlashCK layer norm implementation."""

import torch
import pytest

from flash_ck.ops.norm import layer_norm_fwd
# from ..common.test_framework import UnifiedTestSuite

x = torch.randn(20, 10).cuda()
normalized_shape = (10,)
weight = torch.randn(10).cuda()
bias = torch.randn(10).cuda()
y = layer_norm_fwd(x, normalized_shape, weight, bias)
print(y)
# def reference_impl(x, normalized_shape, weight, bias, eps):
#     """Reference implementation of layer norm for testing."""
#     return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)


# def flashck_impl(x, normalized_shape, weight, bias, eps):
#     """FlashCK implementation of layer norm."""
#     return layer_norm_fwd(x, normalized_shape, weight, bias, eps)


# @pytest.mark.parametrize('shape', [(1024, 4096), (256, 1024)])
# @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
# def test_layernorm_correctness_and_perf(shape, dtype):
#     m, n = shape
#     x = torch.randn(m, n, device='cuda', dtype=dtype)
#     weight = torch.randn(n, device='cuda', dtype=dtype)
#     bias = torch.randn(n, device='cuda', dtype=dtype)
#     op_args = {
#         'x': x,
#         'normalized_shape': (n,),
#         'weight': weight,
#         'bias': bias,
#         'eps': 1e-5,
#     }
#     config = {
#         'name': f'LayerNorm_{m}x{n}_{str(dtype)}',
#         'operation_type': 'LayerNorm',
#         'op_args': op_args,
#     }
#     suite = UnifiedTestSuite(dtype=dtype, device='cuda')
#     suite.run_test([config], reference_impl, flashck_impl,
#                    benchmark=True, num_runs=50, warmup_runs=10, verbose=True)
