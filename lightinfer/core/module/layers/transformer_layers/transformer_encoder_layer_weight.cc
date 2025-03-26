#include "lightinfer/core/module/layers/transformer_layers/transformer_encoder_layer_weight.h"

#include "lightinfer/core/utils/memory_utils.h"

namespace lightinfer {

template<typename T>
TransformerEncoderLayerWeight<T>::TransformerEncoderLayerWeight(size_t hidden_units,
                                                                size_t inter_size,
                                                                size_t tensor_para_size,
                                                                size_t tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    MallocWeights();
    SetWeightPtr();
}

template<typename T>
TransformerEncoderLayerWeight<T>::~TransformerEncoderLayerWeight()
{
    for (auto it = weights_ptr_.begin(); it != weights_ptr_.end(); ++it) {
        DeviceFree(it->second.ptr_);
    }
    weights_ptr_.clear();

    attention_weights_.query_key_value_weight_.kernel_  = nullptr;
    attention_weights_.query_key_value_weight_.bias_    = nullptr;
    attention_weights_.attention_output_weight_.kernel_ = nullptr;
    attention_weights_.attention_output_weight_.bias_   = nullptr;
    attn_layernorm_weights_.gamma_                      = nullptr;
    attn_layernorm_weights_.beta_                       = nullptr;
    ffn_weights_.intermediate_weight_.kernel_           = nullptr;
    ffn_weights_.intermediate_weight_.bias_             = nullptr;
    ffn_weights_.output_weight_.kernel_                 = nullptr;
    ffn_weights_.output_weight_.bias_                   = nullptr;
    ffn_layernorm_weights_.gamma_                       = nullptr;
    ffn_layernorm_weights_.beta_                        = nullptr;
}

template<typename T>
TransformerEncoderLayerWeight<T>::TransformerEncoderLayerWeight(const TransformerEncoderLayerWeight& other):
    TransformerEncoderLayerWeight(other.hidden_units_, other.inter_size_)
{
    for (auto it = other.weights_ptr_.begin(); it != other.weights_ptr_.end(); ++it) {
        HipD2DCpy(weights_ptr_.at(it->first).ptr_, it->second.ptr_, it->second.size_);
    }
}

template<typename T>
TransformerEncoderLayerWeight<T>&
TransformerEncoderLayerWeight<T>::operator=(const TransformerEncoderLayerWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_   = other.inter_size_;

    for (auto it = other.weights_ptr_.begin(); it != other.weights_ptr_.end(); ++it) {
        weights_ptr_.insert({it->first, it->second});
        weights_ptr_.at(it->first).ptr_ = nullptr;
        DeviceMalloc(&weights_ptr_.at(it->first).ptr_, it->second.size_);
        HipD2DCpy(weights_ptr_.at(it->first).ptr_, it->second.ptr_, it->second.size_);
    }
    SetWeightPtr();

    return *this;
}

template<typename T>
void TransformerEncoderLayerWeight<T>::LoadModel(std::string dir_path, DataType model_file_type)
{
    for (auto it = weights_ptr_.begin(); it != weights_ptr_.end(); ++it) {
        LoadWeightFromBin<T>(it->second.ptr_, it->second.shape_, dir_path + it->first, model_file_type);
    }
}

template<typename T>
void TransformerEncoderLayerWeight<T>::LoadParamsPtr(T* qkv_weight,
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
                                                     T* ffn_bias_2)
{
    attention_weights_.query_key_value_weight_.kernel_  = qkv_weight;
    attention_weights_.query_key_value_weight_.bias_    = qkv_bias;
    attention_weights_.attention_output_weight_.kernel_ = out_weight;
    attention_weights_.attention_output_weight_.bias_   = out_bias;
    attn_layernorm_weights_.gamma_                      = layer_norm_gamma_1;
    attn_layernorm_weights_.beta_                       = layer_norm_beta_1;
    ffn_weights_.intermediate_weight_.kernel_           = ffn_weight_1;
    ffn_weights_.intermediate_weight_.bias_             = ffn_bias_1;
    ffn_weights_.output_weight_.kernel_                 = ffn_weight_2;
    ffn_weights_.output_weight_.bias_                   = ffn_bias_2;
    ffn_layernorm_weights_.gamma_                       = layer_norm_gamma_2;
    ffn_layernorm_weights_.beta_                        = layer_norm_beta_2;
}

template<typename T>
void TransformerEncoderLayerWeight<T>::SetWeightPtr()
{
    attention_weights_.query_key_value_weight_.kernel_ =
        weights_ptr_.at("attention.self.query_key_value.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights_.query_key_value_weight_.bias_ =
        weights_ptr_.at("attention.self.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights_.attention_output_weight_.kernel_ = attention_weights_.attention_output_weight_.kernel_ =
        weights_ptr_.at("attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights_.attention_output_weight_.bias_ = weights_ptr_.at("attention.output.dense.bias.bin").ptr_;
    attn_layernorm_weights_.gamma_                    = weights_ptr_.at("attention.output.LayerNorm.weight.bin").ptr_;
    attn_layernorm_weights_.beta_                     = weights_ptr_.at("attention.output.LayerNorm.bias.bin").ptr_;
    ffn_weights_.intermediate_weight_.kernel_ =
        weights_ptr_.at("intermediate.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    ffn_weights_.intermediate_weight_.bias_ =
        weights_ptr_.at("intermediate.dense.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    ffn_weights_.output_weight_.kernel_ =
        weights_ptr_.at("output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    ffn_weights_.output_weight_.bias_ = weights_ptr_.at("output.dense.bias.bin").ptr_;
    ffn_layernorm_weights_.gamma_     = weights_ptr_.at("output.LayerNorm.weight.bin").ptr_;
    ffn_layernorm_weights_.beta_      = weights_ptr_.at("output.LayerNorm.bias.bin").ptr_;
}

template<typename T>
void TransformerEncoderLayerWeight<T>::MallocWeights()
{
    std::string name;
    name = "attention.self.query_key_value.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, Weight<T>(name, {3 * hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.self.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, Weight<T>(name, {3 * hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.output.dense.bias.bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_}, nullptr)});
    name = "attention.output.LayerNorm.weight.bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_}, nullptr)});
    name = "attention.output.LayerNorm.bias.bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_}, nullptr)});
    name = "intermediate.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_, inter_size_ / tensor_para_size_}, nullptr)});
    name = "intermediate.dense.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, Weight<T>(name, {inter_size_ / tensor_para_size_}, nullptr)});
    name = "output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, Weight<T>(name, {inter_size_ / tensor_para_size_, hidden_units_}, nullptr)});
    name = "output.dense.bias.bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_}, nullptr)});
    name = "output.LayerNorm.weight.bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_}, nullptr)});
    name = "output.LayerNorm.bias.bin";
    weights_ptr_.insert({name, Weight<T>(name, {hidden_units_}, nullptr)});

    for (auto it = weights_ptr_.begin(); it != weights_ptr_.end(); ++it) {
        DeviceMalloc(&it->second.ptr_, it->second.size_);
    }
}

template struct TransformerEncoderLayerWeight<float>;
template struct TransformerEncoderLayerWeight<_Float16>;

}  // namespace lightinfer