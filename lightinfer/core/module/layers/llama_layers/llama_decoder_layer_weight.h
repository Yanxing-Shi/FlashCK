#pragma once

#include "lightinfer/core/module/layers/attention_layers/attention_weight.h"
#include "lightinfer/core/module/layers/llama_layers/llama_mlp_weight.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_weight.h"

#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

template<typename T>
struct LlamaDecoderLayerWeight {
public:
    LlamaDecoderLayerWeight() = delete;
    LlamaDecoderLayerWeight(size_t hidden_units,
                            size_t inter_size,
                            size_t tensor_para_size = 1,
                            size_t tensor_para_rank = 0);
    ~LlamaDecoderLayerWeight();
    LlamaDecoderLayerWeight(const LlamaDecoderLayerWeight& other)            = delete;
    LlamaDecoderLayerWeight& operator=(const LlamaDecoderLayerWeight& other) = delete;

    void LoadModel(std::string dir_path, DataType model_file_type);

    LayerNormWeight<T> pre_layernorm_weights_;
    AttentionWeight<T> self_attention_weights_;

    LayerNormWeight<T> post_attention_layernorm_weights_;
    LlamaMLPWeight<T>  mlp_weights_;

private:
    size_t hidden_units_;
    size_t inter_size_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;

    std::vector<std::pair<size_t, T*>> weights_ptr_;

    void SetWeightPtr();
    void MallocWeights();
};

}  // namespace lightinfer