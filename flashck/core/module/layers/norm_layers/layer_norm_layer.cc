#include "flashck/core/module/layers/norm_layers/layer_norm_layer.h"

namespace flashck {

template<typename T>
LayerNormLayer<T>::LayerNormLayer(Shape          normalized_shape,
                                  NormBiasEnum   is_add_bias,
                                  FusedAddEnum   fused_add,
                                  FusedQuantEnum fused_quant):
    Layer("LayerNormLayer")
{
    gamma_var_ = std::make_unique<Variable>("weight_var", CppTypeToDataType<T>::Type());
    beta_var_  = std::make_unique<Variable>("bias_var", CppTypeToDataType<T>::Type());

    gamma_var_->SetShape(normalized_shape);
    beta_var_->SetShape(normalized_shape);

    layer_norm_op_ = std::make_unique<LayerNormOp<T>>(normalized_shape, is_add_bias, fused_add, fused_quant);

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* LayerNormLayer<T>::operator()(Variable* x,
                                        float     eps,
                                        Variable* x_bias,
                                        Variable* x_residual,
                                        Variable* smooth_scale,
                                        Variable* y_residual,
                                        Variable* y_scale)
{
    SetInputs({x});

    Variable* y = (*layer_norm_op_)(
        x, gamma_var_.get(), beta_var_.get(), x_bias, x_residual, smooth_scale, y_residual, y_scale, eps);

    SetOutputs({y});
    return y;
}

template<typename T>
void LayerNormLayer<T>::LoadParam(const T* gamma_ptr, const T* beta_ptr)
{
    gamma_var_->SetValue((char*)gamma_ptr);
    beta_var_->SetValue((char*)beta_ptr);
}

template class LayerNormLayer<float>;
template class LayerNormLayer<_Float16>;
template class LayerNormLayer<ushort>;

}  // namespace flashck