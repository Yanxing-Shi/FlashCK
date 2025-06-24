#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_mutiply_op.h"

namespace flashck {

template<typename T>
GemmRCRBiasMultiplyOp<T>::GemmRCRBiasMultiplyOp(std::string op_name):
    GemmRCRBiasBroadcastOp<T>::GemmRCRBiasBroadcastOp(op_name)
{
    this->op_name_     = "gemm_rcr_bias_multiply";
    this->op_kind_     = GemmOperationKind::Gemm;
    this->num_sources_ = 1;
    this->epilogue_op_ = TensorOperation::AddMultiply;
}

template class GemmRCRBiasMultiplyOp<float>;
template class GemmRCRBiasMultiplyOp<_Float16>;
template class GemmRCRBiasMultiplyOp<ushort>;

}  // namespace flashck