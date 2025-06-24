#include "flashck/core/module/layers/transformer_layers/feed_forward_layer.h"

namespace flashck {

template<typename T>
FeedForwardLayer<T>::FeedForwardLayer(int64_t       hidden_units,
                                      int64_t       inter_size,
                                      LayerNormType layer_norm_type,
                                      float         epsilon):
    Layer("FeedForwardLayer"),
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    layer_norm_type_(layer_norm_type),
    epsilon_(epsilon)
{
    if (layer_norm_type_ == LayerNormType::PreLayerNorm) {
        layer_norm_ = std::make_unique<LayerNormLayer<T>>(Shape({hidden_units_}), epsilon_);
    }

    intermediate_linear_ = std::make_unique<LinearLayer<T>>(hidden_units_, inter_size_, false, true, "gelu");

    output_linear_ = std::make_unique<LinearLayer<T>>(inter_size_, hidden_units_, false, true, "add");

    if (layer_norm_type_ == LayerNormType::PostLayerNorm) {
        layer_norm_ = std::make_unique<LayerNormLayer<T>>(Shape({hidden_units_}), epsilon_);
    }

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* FeedForwardLayer<T>::operator()(Variable* in)
{
    SetInputs({in});

    Variable* residual = in;

    Variable* output = nullptr;
    if (layer_norm_type_ == LayerNormType::PreLayerNorm) {
        Variable* norm_out         = (*layer_norm_)(in);
        Variable* intermediate_out = (*intermediate_linear_)(norm_out);
        output                     = (*output_linear_)(intermediate_out, residual);
    }
    else {
        Variable* intermediate_out = (*intermediate_linear_)(in);
        Variable* output_out       = (*output_linear_)(intermediate_out, residual);
        output                     = (*layer_norm_)(output_out);
    }

    SetOutputs({output});
    return output;
}

template<typename T>
void FeedForwardLayer<T>::LoadParam(const T* gamma_ptr,
                                    const T* beta_ptr,
                                    const T* intermediate_weight_ptr,
                                    const T* intermediate_bias_ptr,
                                    const T* output_weight_ptr,
                                    const T* output_bias_ptr)
{
    layer_norm_->LoadParam(gamma_ptr, beta_ptr);
    intermediate_linear_->LoadParam(intermediate_weight_ptr, intermediate_bias_ptr);
    output_linear_->LoadParam(output_weight_ptr, output_bias_ptr);
}

template class FeedForwardLayer<float>;
template class FeedForwardLayer<_Float16>;

}  // namespace flashck