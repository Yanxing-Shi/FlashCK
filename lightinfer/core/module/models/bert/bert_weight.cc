#include "lightinfer/core/module/models/bert/bert_weight.h"

#include "lightinfer/core/utils/file_utils.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/memory_utils.h"

namespace lightinfer {

template<typename T>
BertWeight<T>::BertWeight(size_t vocab_size,
                          size_t token_type_vocab_size,
                          size_t max_position_embeddings,
                          size_t hidden_units,
                          size_t inter_size,
                          size_t num_layers,
                          size_t tensor_para_size,
                          size_t tensor_para_rank,
                          size_t layer_para_size,
                          size_t layer_para_rank):
    vocab_size_(vocab_size),
    token_type_vocab_size_(token_type_vocab_size),
    max_position_embeddings_(max_position_embeddings),
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    num_layers_(num_layers),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank),
    weights_ptr_(7)
{
    MallocWeights();
    SetWeightPtr();

    bert_encoder_layer_weights_.clear();
    bert_encoder_layer_weights_.reserve(num_layers_);
    for (int i = 0; i < num_layers_; i++) {
        bert_encoder_layer_weights_.push_back(
            new TransformerEncoderLayerWeight<T>(hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_));
    }
}

template<typename T>
BertWeight<T>::~BertWeight()
{
    bert_encoder_layer_weights_.clear();
    for (uint i = 0; i < weights_ptr_.size(); i++) {
        DeviceFree(weights_ptr_[i].second);
        weights_ptr_[i].first = 0;
    }

    bert_embeddings_.word_embeddings_.kernel_       = nullptr;
    bert_embeddings_.token_type_embeddings_.kernel_ = nullptr;
    bert_embeddings_.position_embeddings_.kernel_   = nullptr;
    bert_embeddings_.layer_norm_weight_.gamma_      = nullptr;
    bert_embeddings_.layer_norm_weight_.beta_       = nullptr;
    bert_pooler_.kernel_                            = nullptr;
    bert_pooler_.bias_                              = nullptr;
}

template<typename T>
BertWeight<T>::BertWeight(const BertWeight& other):
    BertWeight(other.hidden_units_,
               other.inter_size_,
               other.num_layers_,
               other.vocab_size_,
               other.token_type_vocab_size_,
               other.max_position_embeddings_)
{

    bert_encoder_layer_weights_.clear();
    bert_encoder_layer_weights_.reserve(num_layers_);
    for (int i = 0; i < num_layers_; i++) {
        bert_encoder_layer_weights_.push_back(other.bert_encoder_layer_weights_[i]);
    }

    for (uint i = 0; i < weights_ptr_.size(); i++) {
        HipD2DCpy(weights_ptr_[i].second, other.weights_ptr_[i].second, weights_ptr_[i].first);
    }
}

template<typename T>
BertWeight<T>& BertWeight<T>::operator=(const BertWeight& other)
{
    hidden_units_            = other.hidden_units_;
    inter_size_              = other.inter_size_;
    num_layers_              = other.num_layers_;
    vocab_size_              = other.vocab_size_;
    token_type_vocab_size_   = other.token_type_vocab_size_;
    max_position_embeddings_ = other.max_position_embeddings_;

    bert_encoder_layer_weights_.clear();
    bert_encoder_layer_weights_.reserve(num_layers_);
    for (int i = 0; i < num_layers_; i++) {
        bert_encoder_layer_weights_.push_back(other.bert_encoder_layer_weights_[i]);
    }

    MallocWeights();

    for (uint i = 0; i < weights_ptr_.size(); i++) {
        HipD2DCpy(weights_ptr_[i].second, other.weights_ptr_[i].second, weights_ptr_[i].first);
    }

    SetWeightPtr();

    return *this;
}

template<typename T>
void BertWeight<T>::LoadModel(std::string dir_path)
{
    DataType model_file_type = GetModelFileType(dir_path + "/config.ini", "bert");
    for (int l = 0; l < num_layers_; l++) {
        bert_encoder_layer_weights_[l]->LoadModel(dir_path + "bert.encoder.layer." + std::to_string(l) + ".",
                                                  model_file_type);
    }

    LoadWeightFromBin(weights_ptr_[0].second,
                      {weights_ptr_[0].first},
                      dir_path + "bert.embeddings.word_embeddings.weight.bin",
                      model_file_type);
    LoadWeightFromBin(weights_ptr_[1].second,
                      {weights_ptr_[1].first},
                      dir_path + "bert.embeddings.token_type_embeddings.weight.bin",
                      model_file_type);
    LoadWeightFromBin(weights_ptr_[2].second,
                      {weights_ptr_[2].first},
                      dir_path + "bert.embeddings.position_embeddings.weight.bin",
                      model_file_type);
    LoadWeightFromBin(weights_ptr_[3].second,
                      {weights_ptr_[3].first},
                      dir_path + "bert.embeddings.LayerNorm.weight.bin",
                      model_file_type);
    LoadWeightFromBin(weights_ptr_[4].second,
                      {weights_ptr_[4].first},
                      dir_path + "bert.embeddings.LayerNorm.bias.bin",
                      model_file_type);

    LoadWeightFromBin(
        weights_ptr_[5].second, {weights_ptr_[5].first}, dir_path + "bert.pooler.dense.weight.bin", model_file_type);
    LoadWeightFromBin(
        weights_ptr_[6].second, {weights_ptr_[6].first}, dir_path + "bert.pooler.dense.bias.bin", model_file_type);
}

template<typename T>
void BertWeight<T>::SetWeightPtr()
{
    bert_embeddings_.word_embeddings_.kernel_       = weights_ptr_[0].second;
    bert_embeddings_.token_type_embeddings_.kernel_ = weights_ptr_[1].second;
    bert_embeddings_.position_embeddings_.kernel_   = weights_ptr_[2].second;
    bert_embeddings_.layer_norm_weight_.gamma_      = weights_ptr_[3].second;
    bert_embeddings_.layer_norm_weight_.beta_       = weights_ptr_[4].second;
    bert_pooler_.kernel_                            = weights_ptr_[5].second;
    bert_pooler_.bias_                              = weights_ptr_[6].second;
}

template<typename T>
void BertWeight<T>::MallocWeights()
{
    weights_ptr_[0].first = vocab_size_ * hidden_units_;               // embeddings.word_embeddings.weight
    weights_ptr_[1].first = token_type_vocab_size_ * hidden_units_;    // embeddings.token_type_embeddings.weight
    weights_ptr_[2].first = max_position_embeddings_ * hidden_units_;  // embeddings.max_position_embeddings.weight
    weights_ptr_[3].first = hidden_units_;                             // embeddings.LayerNorm.weight
    weights_ptr_[4].first = hidden_units_;                             // embeddings.LayerNorm.bias
    weights_ptr_[5].first = hidden_units_ * hidden_units_;             // pooler.dense.weight
    weights_ptr_[6].first = hidden_units_;                             // pooler.dense.bias

    for (uint i = 0; i < weights_ptr_.size(); i++) {
        DeviceMalloc(&weights_ptr_[i].second, weights_ptr_[i].first);
    }
}

template<typename T>
bool BertWeight<T>::IsValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layers_ * 1.0f / layer_para_size_));
    return l < num_layers_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template class BertWeight<float>;
template class BertWeight<_Float16>;

}  // namespace lightinfer
