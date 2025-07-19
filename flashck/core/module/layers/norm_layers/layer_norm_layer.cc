#include "flashck/core/module/layers/norm_layers/layer_norm_layer.h"

namespace flashck {

template<typename T>
LayerNormLayer<T>::LayerNormLayer(Shape          normalized_shape,
                                  NormBiasEnum   is_add_bias,
                                  FusedAddEnum   fused_add,
                                  FusedQuantEnum fused_quant):
    Layer("LayerNormLayer")
{
    // Create learnable parameters with proper data types
    gamma_var_ = std::make_unique<Variable>("weight_var", CppTypeToDataType<T>::value);
    beta_var_  = std::make_unique<Variable>("bias_var", CppTypeToDataType<T>::value);

    // Set parameter shapes to match normalization dimensions
    gamma_var_->SetShape(normalized_shape);
    beta_var_->SetShape(normalized_shape);

    // Initialize the underlying LayerNorm operation
    layer_norm_op_ = std::make_unique<LayerNormOp<T>>(normalized_shape, is_add_bias, fused_add, fused_quant);

    // Finalize layer construction
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
    // Register input tensor for graph building
    SetInputs({x});

    // Execute LayerNorm operation with all optional parameters
    Variable* y = (*layer_norm_op_)(
        x, gamma_var_.get(), beta_var_.get(), x_bias, x_residual, smooth_scale, y_residual, y_scale, eps);

    // Register output tensor for graph building
    SetOutputs({y});
    return y;
}

template<typename T>
void LayerNormLayer<T>::LoadParam(const T* gamma_ptr, const T* beta_ptr)
{
    // Load parameters from host memory (cast to char* for generic pointer handling)
    gamma_var_->SetValue(reinterpret_cast<char*>(const_cast<T*>(gamma_ptr)));
    beta_var_->SetValue(reinterpret_cast<char*>(const_cast<T*>(beta_ptr)));
}

// Explicit template instantiations for supported data types
template class LayerNormLayer<float>;     ///< Single precision floating point
template class LayerNormLayer<_Float16>;  ///< Half precision floating point
template class LayerNormLayer<ushort>;    ///< BFloat16 as unsigned short

}  // namespace flashck