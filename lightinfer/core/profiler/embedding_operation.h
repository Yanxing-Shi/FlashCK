#pragma once

#include <string>

#include "lightinfer/core/profiler/library.h"
#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

class EmbeddingProblem {
public:
    EmbeddingProblem() = default;

    EmbeddingProblem(int64_t         num_indices,
                     int64_t         vocab_size,
                     int64_t         type_vocab_size,
                     int64_t         max_position_embeddings,
                     int64_t         embedding_dims,
                     DataType        emb_dtype,
                     DataType        index_dtype,
                     DataType        gamma_dtype,
                     DataType        beta_dtype,
                     DataType        acc_dtype,
                     DataType        y_dtype,
                     TensorOperation epilogue_op):
        num_indices_(num_indices),
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
        epilogue_op_(epilogue_op)
    {
    }

    EmbeddingProblem(int64_t         num_indices,
                     int64_t         vocab_size,
                     int64_t         embedding_dims,
                     DataType        emb_dtype,
                     DataType        index_dtype,
                     DataType        gamma_dtype,
                     DataType        beta_dtype,
                     DataType        acc_dtype,
                     DataType        y_dtype,
                     TensorOperation epilogue_op):
        num_indices_(num_indices),
        vocab_size_(vocab_size),
        embedding_dims_(embedding_dims),
        emb_dtype_(emb_dtype),
        index_dtype_(index_dtype),
        gamma_dtype_(gamma_dtype),
        beta_dtype_(beta_dtype),
        acc_dtype_(acc_dtype),
        y_dtype_(y_dtype),
        epilogue_op_(epilogue_op)
    {
    }

    int64_t num_indices_;

    int64_t vocab_size_;
    int64_t type_vocab_size_;
    int64_t max_position_embeddings_;
    int64_t embedding_dims_;

    DataType emb_dtype_;
    DataType index_dtype_;
    DataType gamma_dtype_;
    DataType beta_dtype_;
    DataType acc_dtype_;
    DataType y_dtype_;

    TensorOperation epilogue_op_;
};

class EmbeddingTileDesc {
public:
    EmbeddingTileDesc() = default;

    EmbeddingTileDesc(int64_t block_size,
                      int64_t dim_cluster_size,
                      int64_t row_cluster_size,
                      int64_t dim_per_block,
                      int64_t row_per_block,
                      int64_t dim_thread_size,
                      int64_t row_vector_size);

    std::string GetConfigName();

    std::string Emit();

    int64_t block_size_;
    int64_t dim_cluster_size_;
    int64_t row_cluster_size_;
    int64_t dim_per_block_;
    int64_t row_per_block_;
    int64_t dim_thread_size_;
    int64_t row_vector_size_;
};

class EmbeddingOperation {
public:
    EmbeddingOperation() = default;

    std::string GetConfigName();

    std::string Emit();

    EmbeddingOperationKind operation_kind_;
    TensorOperation        epilogue_op_;
    EmbeddingKernelType    embedding_kernel_type_;

    int64_t vocab_size_;
    int64_t type_vocab_size_;
    int64_t max_position_embeddings_;
    int64_t embedding_dims_;

    int64_t num_elements_;

    DataType emb_dtype_;
    DataType index_dtype_;
    DataType gamma_dtype_;
    DataType beta_dtype_;
    DataType acc_dtype_;
    DataType y_dtype_;

    EmbeddingTileDesc tile_desc_;
};

}  // namespace lightinfer