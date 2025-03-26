#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_silu_op.h"

namespace lightinfer {

template<typename T>
GemmRCRBiasSiLUOp<T>::GemmRCRBiasSiLUOp(): GemmRCRBiasOp<T>::GemmRCRBiasOp("gemm_rcr_bias_silu")
{
    this->op_name_     = "gemm_rcr_bias_silu";
    this->op_kind_     = GemmOperationKind::Gemm;
    this->layout_      = DataLayout::RCR;
    this->epilogue_op_ = TensorOperation::AddSiLU;
}

template class GemmRCRBiasSiLUOp<float>;
template class GemmRCRBiasSiLUOp<_Float16>;
template class GemmRCRBiasSiLUOp<ushort>;

}  // namespace lightinfer