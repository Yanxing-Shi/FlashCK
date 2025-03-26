#pragma once

#include <vector>

#include "lightinfer/core/module/layers/embedding_layers/embedding_weight.h"
#include "lightinfer/core/module/layers/gemm_layers/linear_weight.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_weight.h"
#include "lightinfer/core/module/layers/transformer_layers/transformer_encoder_layer_weight.h"

namespace lightinfer {
template<typename T>
class BertWeight {
public:
    BertWeight(size_t vocab_size,
               size_t token_type_vocab_size,
               size_t max_position_embeddings,
               size_t hidden_units,
               size_t inter_size,
               size_t num_layers,
               size_t tensor_para_size = 1,
               size_t tensor_para_rank = 0,
               size_t layer_para_size  = 1,
               size_t layer_para_rank  = 0);
    ~BertWeight();
    BertWeight(const BertWeight& other);
    BertWeight& operator=(const BertWeight& other);

    // Load weight
    void LoadModel(std::string dir_path);

    // bert weight
    std::vector<TransformerEncoderLayerWeight<T>*> bert_encoder_layer_weights_;
    EmbeddingWeight<T>                             bert_embeddings_;
    LinearWeight<T>                                bert_pooler_;

private:
    void SetWeightPtr();
    void MallocWeights();

    bool IsValidLayerParallelId(int l);

    size_t vocab_size_;
    size_t token_type_vocab_size_;
    size_t max_position_embeddings_;

    size_t hidden_units_;
    size_t inter_size_;
    size_t num_layers_;

    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t layer_para_size_;
    size_t layer_para_rank_;

    std::vector<std::pair<size_t, T*>> weights_ptr_;
};

}  // namespace lightinfer