#pragma once

#include "flashck/core/profiling/codegen/codegen_base.h"

namespace flashck {

class EmbeddingQueryEntry: public EntryBase<EmbeddingQueryEntry> {
public:
    std::string exec_entry_;

    int64_t num_embeddings_;
    int64_t vocab_size_;
    int64_t type_vocab_size_;
    int64_t max_position_embeddings_;
    int64_t embedding_dims_;

    std::string emb_dtype_;
    std::string index_dtype_;
    std::string gamma_dtype_;
    std::string beta_dtype_;
    std::string acc_dtype_;
    std::string y_dtype_;

    std::string op_kind_;
    std::string device_;
};

class EmbeddingRecordEntry: public EntryBase<EmbeddingRecordEntry> {
public:
    std::string exec_entry_;

    int64_t num_embeddings_;
    int64_t vocab_size_;
    int64_t type_vocab_size_;
    int64_t max_position_embeddings_;
    int64_t embedding_dims_;

    std::string emb_dtype_;
    std::string index_dtype_;
    std::string gamma_dtype_;
    std::string beta_dtype_;
    std::string acc_dtype_;
    std::string y_dtype_;

    std::string op_kind_;
    std::string device_;

    std::string algo_;
};

}  // namespace flashck