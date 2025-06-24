#pragma once

#include "flashck/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_permute_m2n3_kernel.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"

namespace flashck {

/*
GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
*/

template<typename T>
class GemmRCRBiasPermuteOp: public GemmRCRBiasOp<T> {
public:
    GemmRCRBiasPermuteOp(Shape permute_shape, std::string permute_layout = "m2n3");
    ~GemmRCRBiasPermuteOp() = default;

    using GemmRCRBiasOp<T>::InferShapeImpl;
    Shape InferTrueShape(Variable* a, Variable* b);

    Variable* operator()(Variable* a, Variable* b, Variable* bias);
};

}  // namespace flashck