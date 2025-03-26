#pragma once

#include "lightinfer/core/module/layers/base_weight.h"

#include "lightinfer/core/module/layers/attention_layers/attention_weight.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_weight.h"
#include "lightinfer/core/module/layers/transformer_layers/feed_forward_layer_weight.h"

#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

template<typename T>
struct TransformerEncoderLayerWeight {
public:
    TransformerEncoderLayerWeight() = default;
    TransformerEncoderLayerWeight(size_t hidden_units,
                                  size_t inter_size,
                                  size_t tensor_para_size = 1,
                                  size_t tensor_para_rank = 0);
    ~TransformerEncoderLayerWeight();

    TransformerEncoderLayerWeight(const TransformerEncoderLayerWeight& other);
    TransformerEncoderLayerWeight& operator=(const TransformerEncoderLayerWeight& other);

    void LoadModel(std::string dir_path, DataType model_file_type);

    void LoadParamsPtr(T* qkv_weight,
                       T* qkv_bias,
                       T* out_weight,
                       T* out_bias,
                       T* layer_norm_gamma_1,
                       T* layer_norm_beta_1,
                       T* layer_norm_gamma_2,
                       T* layer_norm_beta_2,
                       T* ffn_weight_1,
                       T* ffn_bias_1,
                       T* ffn_weight_2,
                       T* ffn_bias_2);

    AttentionWeight<T> attention_weights_;
    LayerNormWeight<T> attn_layernorm_weights_;

    FeedForwardWeight<T> ffn_weights_;
    LayerNormWeight<T>   ffn_layernorm_weights_;

private:
    size_t hidden_units_;
    size_t inter_size_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;

    std::unordered_map<std::string, Weight<T>> weights_ptr_;

    void SetWeightPtr();
    void MallocWeights();
};

}  // namespace lightinfer