#pragma once

#include <string>

#include "lightinfer/core/module/layers/embedding_layers/embedding_layer.h"
#include "lightinfer/core/module/layers/gemm_layers/linear_layer.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_layer.h"
#include "lightinfer/core/module/layers/transformer_layers/transformer_encoder_layer.h"

#include "lightinfer/core/module/models/bert/bert_weight.h"

#include "lightinfer/core/module/models/bert/bert_model_utils.h"
#include "lightinfer/core/module/models/model_registry.h"

namespace lightinfer {
template<typename T>
class Bert: public ModelBase {
public:
    Bert(INIReader reader);

    void BuildGraph();

    ~Bert() = default;

    void SetInput(const FeedDataMap& input_data_map);

    void SetOutput(const FeedDataMap& output_data_map);

    FeedDataMap GetOutputData();

    void Forward(hipStream_t stream = nullptr, bool graph_mode = false);

private:
    ModelConfig   model_config_;
    RequestConfig request_config_;

    std::shared_ptr<Context> context_ptr_;

    FeedDataMap output_data_map_;

    // input variables
    std::unique_ptr<Variable> input_ids_;
    std::unique_ptr<Variable> token_type_ids_;
    std::unique_ptr<Variable> position_ids_;

    std::unique_ptr<Variable> bert_out_;

    // layers
    // embedding
    std::unique_ptr<EmbeddingLayer<T>> embedding_layer_;
    // encoder layer
    std::vector<std::unique_ptr<TransformerEncoderLayer<T>>> bert_layer_vec_;
    // pool layer
    std::unique_ptr<LinearLayer<T>> pooler_layer_;

    // weight
    std::unique_ptr<BertWeight<T>> bert_weights_;
};
}  // namespace lightinfer
