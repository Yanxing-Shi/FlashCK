#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "3rdparty/INIReader.h"

namespace lightinfer {

struct ModelConfig {
    std::string model_name_;
    std::string model_dir_;

    int tensor_para_size_;
    int pipeline_para_size_;

    int   vocab_size_;
    int   type_vocab_size_;
    int   max_position_embeddings_;
    int   num_heads_;
    int   size_per_head_;
    int   num_layers_;
    float layernorm_eps_;
    int   hidden_units_;
    int   inter_size_;

    int max_batch_size_;
    int max_seq_len_;
};

struct RequestConfig {
    int request_batch_size_;
    int request_seq_len_;
};

inline ModelConfig ReadModelConfig(const INIReader& reader)
{
    ModelConfig config;

    config.model_name_ = reader.Get("instance_hyperparameter", "model_name");
    config.model_dir_  = std::string(reader.Get("instance_hyperparameter", "model_dir"));

    config.max_batch_size_ = reader.GetInteger("instance_hyperparameter", "max_batch_size");
    config.max_seq_len_    = reader.GetInteger("instance_hyperparameter", "max_seq_len");

    config.tensor_para_size_   = reader.GetInteger("instance_hyperparameter", "tensor_para_size");
    config.pipeline_para_size_ = reader.GetInteger("instance_hyperparameter", "pipeline_para_size");

    // check
    LI_ENFORCE_EQ(
        config.num_heads_ % config.tensor_para_size_,
        0,
        Unavailable("num_heads must be divisible by tensor_para_size, but got num_heads = {}, tensor_para_size = {}",
                    config.num_heads_,
                    config.tensor_para_size_));
    LI_ENFORCE_EQ(config.num_layers_ % config.pipeline_para_size_,
                  0,
                  Unavailable("num_layers must be divisible by pipeline_para_size, but got num_layers = {}, "
                              "pipeline_para_size = {}",
                              config.num_layers_,
                              config.pipeline_para_size_));
    config.vocab_size_              = reader.GetInteger(config.model_name_, "vocab_size");
    config.type_vocab_size_         = reader.GetInteger(config.model_name_, "type_vocab_size");
    config.max_position_embeddings_ = reader.GetInteger(config.model_name_, "max_position_embeddings");
    config.num_heads_               = reader.GetInteger(config.model_name_, "num_heads");
    config.size_per_head_           = reader.GetInteger(config.model_name_, "size_per_head");
    config.num_layers_              = reader.GetInteger(config.model_name_, "num_layers");
    config.layernorm_eps_           = reader.GetFloat(config.model_name_, "layer_norm_eps");

    config.hidden_units_ = config.num_heads_ * config.size_per_head_;
    config.inter_size_   = reader.GetInteger(config.model_name_, "inter_size");

    return config;
}

inline RequestConfig ReadRequestConfig(const INIReader& reader)
{
    RequestConfig config;

    config.request_batch_size_ = reader.GetInteger("request", "request_batch_size");
    config.request_seq_len_    = reader.GetInteger("request", "request_seq_len");

    return config;
}

}  // namespace lightinfer