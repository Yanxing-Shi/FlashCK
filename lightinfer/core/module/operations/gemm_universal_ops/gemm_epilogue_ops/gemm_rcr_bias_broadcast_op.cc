#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_broadcast_op.h"

namespace lightinfer {

template<typename T>
void GemmRCRBiasBroadcastOp<T>::IsVaildInputs(const std::vector<Variable*>& input_var)
{
    LI_ENFORCE_GE(
        input_var.size(),
        3,
        InvalidArgument("input for gemm_rcr_bias_broadcast should be at least 3, got {} instead.", input_var.size()));

    if (input_var.size() > 3) {
        Shape base_shape = this->InferShape(input_var[0], input_var[1]);
        for (int i = 3; i < input_var.size(); i++) {
            Shape d_shape = input_var[i]->GetShape();
            if (d_shape != base_shape) {
                LI_THROW(InvalidArgument("Additional elementwise shape {} doesn't match gemm_bias' shape {}",
                                         d_shape.ToString(),
                                         base_shape.ToString()));
            }
        }
    }
}

template<typename T>
Variable* GemmRCRBiasBroadcastOp<T>::operator()(Variable* a, Variable* b, Variable* bias, Variable* d0)
{
    this->AlignAB(a, b);
    this->input_var_ = {a, b, bias, d0};
    this->IsVaildInputs(this->input_var_);
    this->SanityCheck(a, b);

    Shape output_shape    = this->InferShape(a, b);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_     = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->SetParentsNode({a, b, bias, d0});
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template class GemmRCRBiasBroadcastOp<float>;
template class GemmRCRBiasBroadcastOp<_Float16>;
template class GemmRCRBiasBroadcastOp<ushort>;

}  // namespace lightinfer