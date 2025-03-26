#pragma once

#include "lightinfer/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_gelu_kernel.h"
#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"

namespace lightinfer {

/*
GEMM Specialization: GELU(GEMM_RCR(A, B) + Bias)
*/

template<typename T>
class GemmRCRBiasGeluOp: public GemmRCRBiasOp<T> {
    /*
    GEMM Specialization: FastGELU(GEMM_RCR(A, B))

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(N, K).cuda().half()

        linear = torch.nn.functional.linear(A, B)
        y = torch.nn.GELU(linear)
    */
public:
    GemmRCRBiasGeluOp();
    ~GemmRCRBiasGeluOp() = default;
};

}  // namespace lightinfer