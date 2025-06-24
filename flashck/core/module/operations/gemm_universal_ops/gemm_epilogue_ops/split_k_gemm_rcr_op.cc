#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/split_k_gemm_rcr_op.h"

#include "flashck/core/utils/enforce.h"

#include "flashck/core/graph/node.h"

namespace flashck {

template<typename T>
SplitKGemmRCROp<T>::SplitKGemmRCROp(std::string op_name): GemmRCROp<T>::GemmRCROp(op_name)
{
    this->op_name_     = op_name;
    this->op_kind_     = GemmOperationKind::SplitKGemm;
    this->epilogue_op_ = TensorOperation::PassThrough;
    this->layout_      = DataLayout::RCR;
}

template class SplitKGemmRCROp<float>;
template class SplitKGemmRCROp<_Float16>;
template class SplitKGemmRCROp<ushort>;

}  // namespace flashck