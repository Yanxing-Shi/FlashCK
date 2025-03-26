#pragma once

#include "lightinfer/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_kernel.h"
#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_op.h"

namespace lightinfer {

/*
GEMM Specialization: GEMM_RCR(A, B) + Bias
*/

template<typename CppType>
class GemmRCRBiasOp: public GemmRCROp<CppType> {
    /*
    GEMM Specialization: GEMM_RCR(A, B) + Bias

        This operator is equivalent to the following pytorch code:

        .. highlight:: python
        .. code-block:: python
            A = torch.randn(M, K).cuda().half()
            B = torch.randn(N, K).cuda().half()
            Bias = torch.randn(N).cuda().half()

            y = torch.nn.functional.linear(A, B, bias=Bias)
    */
public:
    GemmRCRBiasOp(std::string op_name = "gemm_rcr_bias");
    ~GemmRCRBiasOp() = default;

    void IsVaildInputs(Variable* a, Variable* b, Variable* bias);

    Variable* operator()(Variable* a, Variable* b, Variable* bias);
};

}  // namespace lightinfer