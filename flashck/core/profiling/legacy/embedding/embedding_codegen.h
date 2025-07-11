#pragma once

#include "flashck/core/profiling/codegen/codegen_base.h"

namespace flashck {

class EmbeddingTileDesc: public TileDescBase {
public:
    std::string GetConfigName() const override;

    std::string Emit() const override;

    int64_t block_size_;
    int64_t dim_cluster_size_;
    int64_t row_cluster_size_;
    int64_t dim_per_block_;
    int64_t row_per_block_;
    int64_t dim_thread_size_;
    int64_t row_vector_size_;
};

class EmbeddingCodegen: public CodegenBase {
public:
    std::string GetConfigName() const override;

    std::string Emit() const override;

    EmbeddingOperationKind kind_;

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

}  // namespace flashck