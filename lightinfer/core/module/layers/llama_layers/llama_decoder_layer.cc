#include "lightinfer/core/module/layers/llama_layers/llama_decoder_layer.h"

namespace lightinfer {

template<typename T>
LlamaDecoderLayer<T>::LlamaDecoderLayer(
    int64_t seq_len, int64_t hidden_dim, int64_t num_heads, int64_t inter_size, float scale):
    Layer("LlamaDecoderLayer")
{
    llama_attn_layer_ = new LlamaAttentionLayer<T>(seq_len, hidden_dim, num_heads, scale);
    llama_mlp_layer_  = new LlamaMLPLayer<T>(hidden_dim, inter_size);

    this->context_ptr_->ExitLayer();  // necessary
}

template<typename T>
Variable* LlamaDecoderLayer<T>::operator()(Variable* input, Variable* cache_k, Variable* cache_v)
{
    SetInputs({
        input,
        cache_k,
        cache_v,
    });

    Variable* attn_out = (*llama_attn_layer_)(input, cache_k, cache_v);

    Variable* mlp_out = (*llama_mlp_layer_)(attn_out);

    SetOutputs({mlp_out});

    return mlp_out;
}

template<typename T>
void LlamaDecoderLayer<T>::LoadParam(const LlamaDecoderLayerWeight<T>* decoder_weight)
{
    llama_attn_layer_->LoadParam(decoder_weight->pre_layernorm_weights_.gamma_,
                                 decoder_weight->pre_layernorm_weights_.beta_,
                                 decoder_weight->self_attention_weights_.query_key_value_weight_.kernel_,
                                 decoder_weight->self_attention_weights_.query_key_value_weight_.bias_,
                                 decoder_weight->self_attention_weights_.attention_output_weight_.kernel_,
                                 decoder_weight->self_attention_weights_.attention_output_weight_.bias_);

    llama_mlp_layer_->LoadParam(decoder_weight->post_attention_layernorm_weights_.gamma_,
                                decoder_weight->post_attention_layernorm_weights_.beta_,
                                decoder_weight->mlp_weights_.gate_weight_.kernel_,
                                decoder_weight->mlp_weights_.gate_weight_.bias_,
                                decoder_weight->mlp_weights_.up_weight_.kernel_,
                                decoder_weight->mlp_weights_.up_weight_.bias_,
                                decoder_weight->mlp_weights_.down_weight_.kernel_,
                                decoder_weight->mlp_weights_.down_weight_.bias_);
}

template class LlamaDecoderLayer<float>;
template class LlamaDecoderLayer<_Float16>;

}  // namespace lightinfer