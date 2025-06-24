#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_tanh_op.h"

namespace flashck {

template<typename T>
GemmRCRBiasTanhOp<T>::GemmRCRBiasTanhOp(): GemmRCRBiasOp<T>::GemmRCRBiasOp("gemm_rcr_bias_tanh")
{
    this->op_name_     = "gemm_rcr_bias_tanh";
    this->op_kind_     = GemmOperationKind::Gemm;
    this->layout_      = DataLayout::RCR;
    this->epilogue_op_ = TensorOperation::AddTanh;
}

template class GemmRCRBiasTanhOp<float>;
template class GemmRCRBiasTanhOp<_Float16>;
template class GemmRCRBiasTanhOp<ushort>;

}  // namespace flashck