#pragma once

#include <string>
#include <unordered_map>
#include <variant>

/*
Embedding profiling cache entries
*/

namespace lightinfer {

// Embedding query entry
struct EmbeddingQueryEntry {
    EmbeddingQueryEntry() = default;

    // general embedding
    EmbeddingQueryEntry(int64_t     num_embeddings,
                        int64_t     embedding_dims,
                        std::string emb_dtype,
                        std::string index_dtype,
                        std::string gamma_dtype,
                        std::string beta_dtype,
                        std::string acc_dtype,
                        std::string y_dtype,
                        std::string op_name,
                        std::string epilogue_op,
                        std::string device,
                        std::string exec_entry_sha1):
        num_embeddings_(num_embeddings),
        vocab_size_(num_embeddings),
        type_vocab_size_(num_embeddings),
        max_position_embeddings_(num_embeddings),
        embedding_dims_(embedding_dims),
        emb_dtype_(emb_dtype),
        index_dtype_(index_dtype),
        gamma_dtype_(gamma_dtype),
        beta_dtype_(beta_dtype),
        acc_dtype_(acc_dtype),
        y_dtype_(y_dtype),
        op_name_(op_name),
        epilogue_op_(epilogue_op),
        device_(device),
        exec_entry_sha1_(exec_entry_sha1)
    {
    }

    // bert embedding
    EmbeddingQueryEntry(int64_t     vocab_size,
                        int64_t     type_vocab_size,
                        int64_t     max_position_embeddings,
                        int64_t     embedding_dims,
                        std::string emb_dtype,
                        std::string index_dtype,
                        std::string gamma_dtype,
                        std::string beta_dtype,
                        std::string acc_dtype,
                        std::string y_dtype,
                        std::string op_name,
                        std::string epilogue_op,
                        std::string device,
                        std::string exec_entry_sha1):
        num_embeddings_(vocab_size),
        vocab_size_(vocab_size),
        type_vocab_size_(type_vocab_size),
        max_position_embeddings_(max_position_embeddings),
        embedding_dims_(embedding_dims),
        emb_dtype_(emb_dtype),
        index_dtype_(index_dtype),
        gamma_dtype_(gamma_dtype),
        beta_dtype_(beta_dtype),
        acc_dtype_(acc_dtype),
        y_dtype_(y_dtype),
        op_name_(op_name),
        epilogue_op_(epilogue_op),
        device_(device),
        exec_entry_sha1_(exec_entry_sha1)
    {
    }

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

    std::string op_name_;
    std::string epilogue_op_;
    std::string device_;
    std::string exec_entry_sha1_;
};

// Profile result record entry
struct EmbeddingRecordEntry {
    EmbeddingRecordEntry() = default;

    // general embedding
    EmbeddingRecordEntry(std::string exec_entry,
                         std::string exec_entry_sha1,
                         int64_t     num_embeddings,
                         int64_t     embedding_dims,
                         std::string emb_dtype,
                         std::string index_dtype,
                         std::string gamma_dtype,
                         std::string beta_dtype,
                         std::string acc_dtype,
                         std::string y_dtype,
                         std::string op_name,
                         std::string epilogue_op,
                         std::string device,
                         std::string algo):
        exec_entry_(exec_entry),
        exec_entry_sha1_(exec_entry_sha1),
        num_embeddings_(num_embeddings),
        vocab_size_(num_embeddings),
        type_vocab_size_(num_embeddings),
        max_position_embeddings_(num_embeddings),
        embedding_dims_(embedding_dims),
        emb_dtype_(emb_dtype),
        index_dtype_(index_dtype),
        gamma_dtype_(gamma_dtype),
        beta_dtype_(beta_dtype),
        acc_dtype_(acc_dtype),
        y_dtype_(y_dtype),
        op_name_(op_name),
        epilogue_op_(epilogue_op),
        device_(device),
        algo_(algo)
    {
    }

    // bert embedding
    EmbeddingRecordEntry(std::string exec_entry,
                         std::string exec_entry_sha1,
                         int64_t     vocab_size,
                         int64_t     type_vocab_size,
                         int64_t     max_position_embeddings,
                         int64_t     embedding_dims,
                         std::string emb_dtype,
                         std::string index_dtype,
                         std::string gamma_dtype,
                         std::string beta_dtype,
                         std::string acc_dtype,
                         std::string y_dtype,
                         std::string op_name,
                         std::string epilogue_op,
                         std::string device,
                         std::string algo):
        exec_entry_(exec_entry),
        exec_entry_sha1_(exec_entry_sha1),
        num_embeddings_(vocab_size),
        vocab_size_(vocab_size),
        type_vocab_size_(type_vocab_size),
        max_position_embeddings_(max_position_embeddings),
        embedding_dims_(embedding_dims),
        emb_dtype_(emb_dtype),
        index_dtype_(index_dtype),
        gamma_dtype_(gamma_dtype),
        beta_dtype_(beta_dtype),
        acc_dtype_(acc_dtype),
        y_dtype_(y_dtype),
        op_name_(op_name),
        epilogue_op_(epilogue_op),
        device_(device),
        algo_(algo)
    {
    }

    std::string exec_entry_;
    std::string exec_entry_sha1_;

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

    std::string op_name_;
    std::string epilogue_op_;
    std::string device_;
    std::string algo_;
};

}  // namespace lightinfer