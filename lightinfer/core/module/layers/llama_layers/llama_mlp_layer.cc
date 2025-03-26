#include "lightinfer/core/module/layers/llama_layers/llama_mlp_layer.h"

namespace lightinfer {

template<typename T>
LlamaMLPLayer<T>::LlamaMLPLayer(
    int64_t hidden_dim, int64_t inter_size, bool use_up_bias, bool use_down_bias, bool use_gate_bias):
    Layer("LlamaMLPLayer"),
    hidden_dim_(hidden_dim),
    inter_size_(inter_size),
    use_up_bias_(use_up_bias),
    use_down_bias_(use_down_bias),
    use_gate_bias_(use_gate_bias)
{
    pre_layer_norm_ = std::make_unique<LayerNormLayer<T>>(Shape({hidden_dim_}));

    gate_proj_ = std::make_unique<LinearLayer<T>>(hidden_dim_, inter_size_, use_gate_bias, "silu");

    up_proj_ = std::make_unique<LinearLayer<T>>(hidden_dim_, inter_size_, use_up_bias, "multiply");

    down_proj_ = std::make_unique<LinearLayer<T>>(inter_size_, hidden_dim_, use_down_bias, "add");

    this->context_ptr_->ExitLayer();
}

template<typename T>
LlamaMLPLayer<T>::~LlamaMLPLayer()
{
}

template<typename T>
Variable* LlamaMLPLayer<T>::operator()(Variable* in)
{
    SetInputs({in});

    Variable* residual = in;

    Variable* norm_out        = (*pre_layer_norm_)(in);
    Variable* norm_out_backup = norm_out;
    Variable* gate_out        = (*gate_proj_)(norm_out);
    Variable* up_out          = (*up_proj_)(norm_out_backup, gate_out);
    Variable* down_out        = (*down_proj_)(up_out, residual);

    SetOutputs({down_out});
    return down_out;
}

template<typename T>
void LlamaMLPLayer<T>::LoadParam(const T* gamma_ptr,
                                 const T* beta_ptr,
                                 const T* up_weight_ptr,
                                 const T* up_bias_ptr,
                                 const T* down_weight_ptr,
                                 const T* down_bias_ptr,
                                 const T* gate_weight_ptr,
                                 const T* gate_bias_ptr)
{
    pre_layer_norm_->LoadParam(gamma_ptr, beta_ptr);
    gate_proj_->LoadParam(gate_weight_ptr, gate_bias_ptr);
    up_proj_->LoadParam(up_weight_ptr, up_bias_ptr);
    down_proj_->LoadParam(down_weight_ptr, down_bias_ptr);
}

template class LlamaMLPLayer<float>;
template class LlamaMLPLayer<_Float16>;

}  // namespace lightinfer