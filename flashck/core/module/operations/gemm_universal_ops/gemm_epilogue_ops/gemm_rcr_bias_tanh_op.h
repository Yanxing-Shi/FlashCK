#pragma once

#include "flashck/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_tanh_kernel.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"

namespace flashck {

/*
GEMM Specialization: SILU(GEMM_RCR(A, B) + Bias)
*/

template<typename T>
class GemmRCRBiasTanhOp: public GemmRCRBiasOp<T> {
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
    GemmRCRBiasTanhOp();
    ~GemmRCRBiasTanhOp() = default;
};

}  // namespace flashck