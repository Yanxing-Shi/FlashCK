#include "flashck/core/module/layers/transformer_layers/transformer_encoder_layer.h"

namespace flashck {

template<typename T>
TransformerEncoderLayer<T>::TransformerEncoderLayer(int64_t         seq_len,
                                                    int64_t         hidden_dim,
                                                    int64_t         num_heads,
                                                    int64_t         inter_size,
                                                    LayerNormType   layer_norm_type,
                                                    float           epsilon,
                                                    TensorOperation mask,
                                                    float           scale):
    Layer("TransformerEncoderLayer")
{
    attn_layer_ = std::make_unique<MultiHeadAttentionLayer<T>>(
        hidden_dim, seq_len, num_heads, scale, mask, layer_norm_type, epsilon);
    feed_forward_layer_ = std::make_unique<FeedForwardLayer<T>>(hidden_dim, inter_size, layer_norm_type, epsilon);

    this->context_ptr_->ExitLayer();  // necessary
}

template<typename T>
Variable* TransformerEncoderLayer<T>::operator()(Variable* input)
{
    SetInputs({input});

    Variable* attn_out = (*attn_layer_)(input);

    Variable* ffn_out = (*feed_forward_layer_)(attn_out);

    SetOutputs({ffn_out});

    return ffn_out;
}

template<typename T>
void TransformerEncoderLayer<T>::LoadParam(TransformerEncoderLayerWeight<T>* encoder_weight)
{
    attn_layer_->LoadParam(encoder_weight->attn_layernorm_weights_.gamma_,
                           encoder_weight->attn_layernorm_weights_.beta_,
                           encoder_weight->attention_weights_.query_key_value_weight_.kernel_,
                           encoder_weight->attention_weights_.query_key_value_weight_.bias_,
                           encoder_weight->attention_weights_.attention_output_weight_.kernel_,
                           encoder_weight->attention_weights_.attention_output_weight_.bias_);

    feed_forward_layer_->LoadParam(encoder_weight->ffn_layernorm_weights_.gamma_,
                                   encoder_weight->ffn_layernorm_weights_.beta_,
                                   encoder_weight->ffn_weights_.intermediate_weight_.kernel_,
                                   encoder_weight->ffn_weights_.intermediate_weight_.bias_,
                                   encoder_weight->ffn_weights_.output_weight_.kernel_,
                                   encoder_weight->ffn_weights_.output_weight_.bias_);
}

template class TransformerEncoderLayer<float>;
template class TransformerEncoderLayer<_Float16>;

}  // namespace flashck