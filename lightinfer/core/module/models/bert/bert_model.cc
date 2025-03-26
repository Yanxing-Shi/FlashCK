#include "lightinfer/core/module/models/bert/bert_model.h"

#include "lightinfer/core/utils/memory_utils.h"

namespace lightinfer {

template<typename T>
Bert<T>::Bert(const INIReader reader)
{
    Context::CreateGlobalContext("bert", Mode::Inference);
    context_ptr_ = Context::GetGlobalInstance();

    model_config_   = ReadModelConfig(reader);
    request_config_ = ReadRequestConfig(reader);
}

template<typename T>
void Bert<T>::BuildGraph()
{
    bert_weights_ = std::make_unique<BertWeight<T>>(model_config_.vocab_size_,
                                                    model_config_.type_vocab_size_,
                                                    model_config_.max_position_embeddings_,
                                                    model_config_.hidden_units_,
                                                    model_config_.inter_size_,
                                                    model_config_.num_layers_);
    bert_weights_->LoadModel(model_config_.model_dir_);

    input_ids_ = std::make_unique<Variable>("input_ids", DataType::INT32);
    input_ids_->SetShape({DDim({1, model_config_.max_batch_size_}), DDim({1, model_config_.max_seq_len_})});

    token_type_ids_ = std::make_unique<Variable>("token_type_ids", DataType::INT32);
    token_type_ids_->SetShape({DDim({1, model_config_.max_batch_size_}), DDim({1, model_config_.max_seq_len_})});

    position_ids_ = std::make_unique<Variable>("position_ids", DataType::INT32);
    position_ids_->SetShape({DDim({1, model_config_.max_batch_size_}), DDim({1, model_config_.max_seq_len_})});

    embedding_layer_ = std::make_unique<EmbeddingLayer<T>>(model_config_.vocab_size_,
                                                           model_config_.type_vocab_size_,
                                                           model_config_.max_position_embeddings_,
                                                           model_config_.hidden_units_);
    embedding_layer_->LoadParam(bert_weights_->bert_embeddings_.word_embeddings_.kernel_,
                                bert_weights_->bert_embeddings_.token_type_embeddings_.kernel_,
                                bert_weights_->bert_embeddings_.position_embeddings_.kernel_,
                                bert_weights_->bert_embeddings_.layer_norm_weight_.gamma_,
                                bert_weights_->bert_embeddings_.layer_norm_weight_.beta_);

    for (int layer_idx = 0; layer_idx < model_config_.num_layers_; layer_idx++) {
        auto bert_layer = std::make_unique<TransformerEncoderLayer<T>>(model_config_.max_seq_len_,
                                                                       model_config_.hidden_units_,
                                                                       model_config_.num_heads_,
                                                                       model_config_.inter_size_);
        bert_layer->LoadParam(bert_weights_->bert_encoder_layer_weights_[layer_idx]);
        bert_layer_vec_.push_back(std::move(bert_layer));
    }

    pooler_layer_ = std::make_unique<LinearLayer<T>>(model_config_.hidden_units_, model_config_.hidden_units_);
    pooler_layer_->LoadParam(bert_weights_->bert_pooler_.kernel_, bert_weights_->bert_pooler_.bias_);

    Variable* emb_outs     = (*embedding_layer_)(input_ids_.get(), token_type_ids_.get(), position_ids_.get());
    Variable* encoder_outs = nullptr;
    for (const auto& layer : bert_layer_vec_) {
        encoder_outs = (*layer)(emb_outs);
    }
    bert_out_.reset((*pooler_layer_)(encoder_outs));

    context_ptr_->CodegenAndProfileKernel();
    context_ptr_->BuildContext();

    LOG(INFO) << "Finish construct network!!";
}

template<typename T>
void Bert<T>::SetInput(const FeedDataMap& input_data_map)
{
    int runtime_batch_size = input_data_map.At("input_ids").shape_.GetDim(0).GetValues()[0];
    int runtime_seq_len    = input_data_map.At("input_ids").shape_.GetDim(1).GetValues()[0];

    LI_ENFORCE_EQ(request_config_.request_batch_size_,
                  runtime_batch_size,
                  Unavailable("Input batch size is not equal to request batch size."));
    LI_ENFORCE_EQ(request_config_.request_seq_len_,
                  runtime_seq_len,
                  Unavailable("Input seq len is not equal to request seq len."));

    LI_ENFORCE_EQ(runtime_batch_size <= model_config_.max_batch_size_ && runtime_seq_len <= model_config_.max_seq_len_,
                  true,
                  Unavailable("Input data ({}, {}) is larger than {} or max input seq len {}.",
                              runtime_batch_size,
                              runtime_seq_len,
                              model_config_.max_batch_size_,
                              model_config_.max_seq_len_));

    input_ids_->SetValue((char*)input_data_map.GetPtr<int>("input_ids"));
    input_ids_->SetShape({runtime_batch_size, runtime_seq_len});

    token_type_ids_->SetValue((char*)input_data_map.GetPtr<int>("token_type_ids"));
    token_type_ids_->SetShape({runtime_batch_size, runtime_seq_len});

    position_ids_->SetValue((char*)input_data_map.GetPtr<int>("position_ids"));
    position_ids_->SetShape({runtime_batch_size, runtime_seq_len});
}

template<typename T>
void Bert<T>::SetOutput(const FeedDataMap& output_data_map)
{
    output_data_map_ = output_data_map;
    bert_out_->SetValue((char*)output_data_map.GetPtr<T>("output_hidden_state"));
}

template<typename T>
FeedDataMap Bert<T>::GetOutputData()
{
    LI_ENFORCE_EQ(bert_out_->GetShape().GetDim(0).GetValues()[0],
                  request_config_.request_batch_size_,
                  Unavailable("Output batch size is not equal to request batch size."));

    LI_ENFORCE_EQ(bert_out_->GetShape().GetDim(1).GetValues()[0],
                  request_config_.request_seq_len_,
                  Unavailable("Output seq len is not equal to request seq len."));

    LI_ENFORCE_EQ(bert_out_->GetShape().GetDim(2).GetValues()[0],
                  model_config_.hidden_units_,
                  Unavailable("Output hidden size is not equal to model hidden size."));

    auto output_value = (T*)bert_out_->GetValue();
    PrintToScreen(output_value, 5);

    auto original_value = output_data_map_.GetPtr<T>("output_hidden_state");
    original_value      = output_value;

    return output_data_map_;
}

template<typename T>
void Bert<T>::Forward(hipStream_t stream, bool graph_mode)
{
    embedding_layer_->Forward();
    for (const auto& layer : bert_layer_vec_) {
        layer->Forward();
    }
    pooler_layer_->Forward();
}

template class Bert<float>;
template class Bert<_Float16>;

}  // namespace lightinfer