#pragma once

#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/graph/node.h"

#include "lightinfer/core/module/layers/attention_layers/multi_head_attention_layer.h"
#include "lightinfer/core/module/layers/transformer_layers/feed_forward_layer.h"

#include "lightinfer/core/module/layers/transformer_layers/transformer_encoder_layer_weight.h"

namespace lightinfer {

template<typename T>
class TransformerEncoderLayer: public Layer {
public:
    TransformerEncoderLayer(int64_t         seq_len,
                            int64_t         hidden_dim,
                            int64_t         num_heads,
                            int64_t         inter_size,
                            LayerNormType   layer_norm_type = LayerNormType::PostLayerNorm,
                            float           epsilon         = 1e-5f,
                            TensorOperation mask            = TensorOperation::MaskDisabled,
                            float           scale           = 1.0f);
    ~TransformerEncoderLayer() = default;

    Variable* operator()(Variable* input);

    void LoadParam(TransformerEncoderLayerWeight<T>* encoder_weight);

    // layer
    std::unique_ptr<MultiHeadAttentionLayer<T>> attn_layer_;
    std::unique_ptr<FeedForwardLayer<T>>        feed_forward_layer_;
};
}  // namespace lightinfer