#pragma once

#include "lightinfer/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_silu_kernel.h"
#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"

namespace lightinfer {

/*
GEMM Specialization: SILU(GEMM_RCR(A, B) + Bias)
*/

template<typename T>
class GemmRCRBiasSiLUOp: public GemmRCRBiasOp<T> {
    /*
    GEMM Specialization: GEMM_RCR(A, B) + Bias

        This operator is equivalent to the following pytorch code:

        .. highlight:: python
        .. code-block:: python
            A = torch.randn(M, K).cuda().half()
            B = torch.randn(N, K).cuda().half()
            Bias = torch.randn(N).cuda().half()

            linear = torch.nn.functional.linear(A, B, bias=Bias)
            y = torch.nn.SiLU(linear)
    */
public:
    GemmRCRBiasSiLUOp();
    ~GemmRCRBiasSiLUOp() = default;
};

}  // namespace lightinfer