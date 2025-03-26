#pragma once

#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/graph/node.h"

#include "lightinfer/core/module/layers/llama_layers/llama_attention_layer.h"
#include "lightinfer/core/module/layers/llama_layers/llama_mlp_layer.h"

#include "lightinfer/core/module/layers/llama_layers/llama_decoder_layer_weight.h"

namespace lightinfer {

template<typename T>
class LlamaDecoderLayer: public Layer {
public:
    LlamaDecoderLayer(int64_t seq_len, int64_t hidden_dim, int64_t num_heads, int64_t inter_size, float scale = 1.0f);
    ~LlamaDecoderLayer() = default;

    Variable* operator()(Variable* input, Variable* cache_k, Variable* cache_v);

    void LoadParam(const LlamaDecoderLayerWeight<T>* decoder_weight);

    // layer
    LlamaAttentionLayer<T>* llama_attn_layer_;
    LlamaMLPLayer<T>*       llama_mlp_layer_;
};
}  // namespace lightinfer