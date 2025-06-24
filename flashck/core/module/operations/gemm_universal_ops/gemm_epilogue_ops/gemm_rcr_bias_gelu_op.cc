#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_gelu_op.h"

namespace flashck {

template<typename T>
GemmRCRBiasGeluOp<T>::GemmRCRBiasGeluOp(): GemmRCRBiasOp<T>::GemmRCRBiasOp("gemm_rcr_bias_gelu")
{
    this->op_name_     = "gemm_rcr_bias_gelu";
    this->op_kind_     = GemmOperationKind::Gemm;
    this->layout_      = DataLayout::RCR;
    this->epilogue_op_ = TensorOperation::AddGelu;
}

template class GemmRCRBiasGeluOp<float>;
template class GemmRCRBiasGeluOp<_Float16>;
template class GemmRCRBiasGeluOp<ushort>;

}  // namespace flashck