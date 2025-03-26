
#include "lightinfer/core/module/layers/llama_layers/llama_decoder_layer_weight.h"

#include "lightinfer/core/utils/memory_utils.h"

namespace lightinfer {

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(size_t hidden_units,
                                                    size_t inter_size,
                                                    size_t tensor_para_size,
                                                    size_t tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    weights_ptr_(14)
{
    MallocWeights();
    SetWeightPtr();
}

template<typename T>
LlamaDecoderLayerWeight<T>::~LlamaDecoderLayerWeight()
{
    for (int i = 0; i < weights_ptr_.size(); i++) {
        DeviceFree(weights_ptr_[i].second);
        weights_ptr_[i].first = 0;
    }

    pre_layernorm_weights_.gamma_ = nullptr;
    pre_layernorm_weights_.beta_  = nullptr;

    self_attention_weights_.query_key_value_weight_.kernel_  = nullptr;
    self_attention_weights_.query_key_value_weight_.bias_    = nullptr;
    self_attention_weights_.attention_output_weight_.kernel_ = nullptr;
    self_attention_weights_.attention_output_weight_.bias_   = nullptr;

    post_attention_layernorm_weights_.beta_  = nullptr;
    post_attention_layernorm_weights_.gamma_ = nullptr;

    mlp_weights_.gate_weight_.kernel_ = nullptr;
    mlp_weights_.gate_weight_.bias_   = nullptr;
    mlp_weights_.up_weight_.kernel_   = nullptr;
    mlp_weights_.up_weight_.bias_     = nullptr;
    mlp_weights_.down_weight_.kernel_ = nullptr;
    mlp_weights_.down_weight_.bias_   = nullptr;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::LoadModel(std::string dir_path, DataType model_file_type)
{
    const std::string rank_spec = std::to_string(tensor_para_rank_);

    // layernorm
    LoadWeightFromBin<T>(
        weights_ptr_[0].second, {weights_ptr_[0].first}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    DeviceFill(weights_ptr_[1].second, {weights_ptr_[1].first}, (T)0.0);

    // llama attn
    LoadWeightFromBin<T>(weights_ptr_[2].second,
                         {weights_ptr_[2].first},
                         dir_path + ".self_attn.q_k_v_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    DeviceFill(weights_ptr_[3].second, {weights_ptr_[3].first}, (T)0.0);

    LoadWeightFromBin<T>(weights_ptr_[4].second,
                         {weights_ptr_[4].first},
                         dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                         model_file_type);
    DeviceFill(weights_ptr_[5].second, {weights_ptr_[5].first}, (T)0.0);

    // llama mlp
    LoadWeightFromBin<T>(weights_ptr_[6].second,
                         {weights_ptr_[6].first},
                         dir_path + ".post_attention_layernorm.weight.bin",
                         model_file_type);
    DeviceFill(weights_ptr_[7].second, {weights_ptr_[7].first}, (T)0.0);

    LoadWeightFromBin<T>(weights_ptr_[8].second,
                         {weights_ptr_[8].first},
                         dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    DeviceFill(weights_ptr_[9].second, {weights_ptr_[9].first}, (T)0.0);

    LoadWeightFromBin<T>(weights_ptr_[10].second,
                         {weights_ptr_[10].first},
                         dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    DeviceFill(weights_ptr_[11].second, {weights_ptr_[11].first}, (T)0.0);

    LoadWeightFromBin<T>(weights_ptr_[12].second,
                         {weights_ptr_[12].first},
                         dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    DeviceFill(weights_ptr_[13].second, {weights_ptr_[13].first}, (T)0.0);
}

template<typename T>
void LlamaDecoderLayerWeight<T>::SetWeightPtr()
{
    pre_layernorm_weights_.beta_  = weights_ptr_[0].second;
    pre_layernorm_weights_.gamma_ = weights_ptr_[1].second;

    self_attention_weights_.query_key_value_weight_.kernel_  = weights_ptr_[2].second;
    self_attention_weights_.query_key_value_weight_.bias_    = weights_ptr_[3].second;
    self_attention_weights_.attention_output_weight_.kernel_ = weights_ptr_[4].second;
    self_attention_weights_.attention_output_weight_.bias_   = weights_ptr_[5].second;

    post_attention_layernorm_weights_.beta_  = weights_ptr_[6].second;
    post_attention_layernorm_weights_.gamma_ = weights_ptr_[7].second;

    mlp_weights_.gate_weight_.kernel_ = weights_ptr_[8].second;
    mlp_weights_.gate_weight_.bias_   = weights_ptr_[9].second;
    mlp_weights_.up_weight_.kernel_   = weights_ptr_[10].second;
    mlp_weights_.up_weight_.bias_     = weights_ptr_[11].second;
    mlp_weights_.down_weight_.kernel_ = weights_ptr_[12].second;
    mlp_weights_.down_weight_.bias_   = weights_ptr_[13].second;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::MallocWeights()
{
    weights_ptr_[0].first = hidden_units_;  // pre layernorm beta
    weights_ptr_[1].first = hidden_units_;  // pre layernorm gamma

    weights_ptr_[2].first = hidden_units_ * 3 * hidden_units_ / tensor_para_size_;  // qkv weight
    weights_ptr_[3].first = 3 * hidden_units_ / tensor_para_size_;                  // qkv bias

    weights_ptr_[4].first = hidden_units_ / tensor_para_size_ * hidden_units_;  // attention output weight
    weights_ptr_[5].first = hidden_units_;                                      // attention output bias

    weights_ptr_[6].first = hidden_units_;  // post attn layernorm beta
    weights_ptr_[7].first = hidden_units_;  // post attn layernorm gamma

    weights_ptr_[8].first = hidden_units_ * inter_size_ / tensor_para_size_;  // gate proj kernel
    weights_ptr_[9].first = inter_size_ / tensor_para_size_;                  // gate proj bias

    weights_ptr_[10].first = hidden_units_ * inter_size_ / tensor_para_size_;  // up proj kernel
    weights_ptr_[11].first = inter_size_ / tensor_para_size_;                  // up proj bias

    weights_ptr_[12].first = inter_size_ / tensor_para_size_ * hidden_units_;  // down proj kernel
    weights_ptr_[13].first = hidden_units_;                                    // down proj bias

    for (int i = 0; i < weights_ptr_.size(); i++) {
        DeviceMalloc(&weights_ptr_[i].second, weights_ptr_[i].first);
    }
}

template struct LlamaDecoderLayerWeight<float>;
template struct LlamaDecoderLayerWeight<_Float16>;

}  // namespace lightinfer
