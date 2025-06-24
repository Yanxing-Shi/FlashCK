#pragma once

#include "flashck/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_multiply_kernel.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_broadcast_op.h"

namespace flashck {

/*
GEMM Specialization: (GEMM_RCR(A, B) + Bias) * D0
*/

template<typename T>
class GemmRCRBiasMultiplyOp: public GemmRCRBiasBroadcastOp<T> {
    /*
    GEMM Specialization: (GEMM_RCR(A, B) + Bias) * D0

        This operator is equivalent to the following pytorch code:

        .. highlight:: python
        .. code-block:: python

            A = torch.randn(M, K).cuda().half()
            B = torch.randn(N, K).cuda().half()
            Bias = torch.randn(N).cuda().half()
            D0 = torch.randn(M, N).cuda().half()

            linear = torch.nn.functional.linear(A, B, bias=Bias)
            y = linear * D0
    */
public:
    GemmRCRBiasMultiplyOp(std::string op_name = "gemm_rcr_bias_multiply_op");
};

}  // namespace flashck