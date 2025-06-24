#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"

#include "flashck/core/utils/enforce.h"

#include "flashck/core/graph/node.h"

namespace flashck {

template<typename T>
GemmRCRBiasOp<T>::GemmRCRBiasOp(std::string op_name): GemmRCROp<T>::GemmRCROp(op_name)
{
    this->op_name_     = op_name;
    this->op_kind_     = GemmOperationKind::Gemm;
    this->layout_      = DataLayout::RCR;
    this->epilogue_op_ = TensorOperation::Add;
}

template<typename T>
void GemmRCRBiasOp<T>::IsVaildInputs(Variable* a, Variable* b, Variable* bias)
{
    Shape bias_shapes = bias->GetShape();
    if (bias_shapes.GetNumDim() != 1)
        LI_THROW(Unavailable("Bias should be 1D vector! Current bias shape: {}", bias_shapes.ToString()));

    // check a and b shape
    Shape a_shape = a->GetShape();
    Shape b_shape = b->GetShape();
    VLOG(1) << "a_shape" << a_shape.ToString();
    VLOG(1) << "b_shape" << b_shape.ToString();

    DDim bias_shape = bias_shapes.GetDim(0);
    VLOG(1) << "bias_shape" << bias_shape.ToString();
    if (!bias_shape.IsStatic()) {
        LI_THROW(Unavailable("Bias should be fixed 1D vector! Current bias shape: {}", bias_shape.ToString()));
    }

    Shape output_shape = this->InferShape(a, b);
    VLOG(1) << "c_shape" << output_shape.ToString();
    if (output_shape.GetLastDim() != bias_shape) {
        LI_THROW(Unavailable("GEMM/Bias shape doesn't match! Gemm shape: {}, bias shape: {}",
                             output_shape.ToString(),
                             bias_shape.ToString()));
    }
}

template<typename T>
Variable* GemmRCRBiasOp<T>::operator()(Variable* a, Variable* b, Variable* bias)
{
    this->AlignAB(a, b);
    this->IsVaildInputs(a, b, bias);
    this->input_var_ = {a, b, bias};
    this->SanityCheck(a, b);

    Shape output_shape    = this->InferShape(a, b);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_     = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->SetParentsNode({a, b, bias});
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template class GemmRCRBiasOp<float>;
template class GemmRCRBiasOp<_Float16>;
template class GemmRCRBiasOp<ushort>;

}  // namespace flashck