#include "flashck/core/module/layers/norm_layers/rms_norm_layer.h"

namespace flashck {

template<typename T>
RMSNormLayer<T>::RMSNormLayer(Shape normalized_shape, float eps, FusedAddEnum fused_add, FusedQuantEnum fused_quant):
    Layer("RMSNormLayer"),
    normalized_shape_(normalized_shape),
    eps_(eps),
    fused_add_(fused_add),
    fused_quant_(fused_quant)
{
    // param node
    gamma_var_ = std::make_unique<Variable>("weight_var", CppTypeToDataType<T>::Type());  // gamma

    gamma_var_->SetShape(normalized_shape_);

    rms_norm_op_ = std::make_unique<RMSNormOp<T>>(normalized_shape_, fused_add_, fused_quant_);

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* RMSNormLayer<T>::operator()(
    Variable* x, Variable* x_residual, Variable* smooth_scale, Variable* y_residual, Variable* y_scale)
{
    SetInputs({x});

    Variable* y =
        (*rms_norm_op_)(x, gamma_var_.get(), x_residual, smooth_scale, y_residual, y_scale, normalized_shape_, eps_);

    SetOutputs({y});
    return y;
}

template<typename T>
void RMSNormLayer<T>::LoadParam(const T* gamma_ptr)
{
    gamma_var_->SetValue((char*)gamma_ptr);
}

template class RMSNormLayer<float>;
template class RMSNormLayer<_Float16>;
template class RMSNormLayer<ushort>;

}  // namespace flashck