#include "flashck/core/module/layers/norm_layers/rms_norm_layer.h"

namespace flashck {

template<typename T>
RMSNormLayer<T>::RMSNormLayer(Shape normalized_shape, float eps, FusedAddEnum fused_add, FusedQuantEnum fused_quant):
    Layer("RMSNormLayer")
{
    // Create learnable scale parameter with proper data type
    gamma_var_ = std::make_unique<Variable>("weight_var", CppTypeToDataType<T>::value);

    // Set parameter shape to match normalization dimensions
    gamma_var_->SetShape(normalized_shape);

    // Initialize the underlying RMSNorm operation
    rms_norm_op_ = std::make_unique<RMSNormOp<T>>(normalized_shape, fused_add, fused_quant);

    // Finalize layer construction
    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* RMSNormLayer<T>::operator()(
    Variable* x, float eps, Variable* x_residual, Variable* smooth_scale, Variable* y_residual, Variable* y_scale)
{
    // Register input tensor for graph building
    SetInputs({x});

    // Execute RMSNorm operation with all optional parameters
    Variable* y = (*rms_norm_op_)(x, gamma_var_.get(), x_residual, smooth_scale, y_residual, y_scale, eps);

    // Register output tensor for graph building
    SetOutputs({y});
    return y;
}

template<typename T>
void RMSNormLayer<T>::LoadParam(const T* gamma_ptr)
{
    // Load parameter from host memory (cast to char* for generic pointer handling)
    gamma_var_->SetValue(reinterpret_cast<char*>(const_cast<T*>(gamma_ptr)));
}

// Explicit template instantiations for supported data types
template class RMSNormLayer<float>;     ///< Single precision floating point
template class RMSNormLayer<_Float16>;  ///< Half precision floating point
template class RMSNormLayer<ushort>;    ///< BFloat16 as unsigned short

}  // namespace flashck