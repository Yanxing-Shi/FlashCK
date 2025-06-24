#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_add_op.h"

namespace flashck {

template<typename T>
GemmRCRBiasAddOp<T>::GemmRCRBiasAddOp(std::string op_name): GemmRCRBiasBroadcastOp<T>::GemmRCRBiasBroadcastOp(op_name)
{
    this->op_name_     = "gemm_rcr_bias_add";
    this->op_kind_     = GemmOperationKind::Gemm;
    this->num_sources_ = 1;
    this->epilogue_op_ = TensorOperation::AddAdd;
}

template class GemmRCRBiasAddOp<float>;
template class GemmRCRBiasAddOp<_Float16>;
template class GemmRCRBiasAddOp<ushort>;

}  // namespace flashck