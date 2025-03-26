#pragma once

#include "lightinfer/core/module/operations/embedding_ops/embedding_common_op.h"

namespace lightinfer {

/*
embedding operation
*/

template<typename T>
class EmbeddingAddAddLayerNormOp: public EmbeddingCommonOp<T, EmbeddingAddAddLayerNormOp<T>> {
public:
    EmbeddingAddAddLayerNormOp(int64_t     vocab_size,
                               int64_t     type_vocab_size,
                               int64_t     max_position_embeddings,
                               int64_t     embedding_dims,
                               float       epsilon = 1e-12,
                               std::string op_name = "embedding_add_add_layer_norm");

    EmbeddingQueryEntry GetEmbeddingQueryEntry(const std::string& workload);

    EmbeddingProblem GetEmbeddingProblem(const int64_t num_indices);

    std::vector<std::string>
    GenOpProfileCmd(const std::string& profiler_prefix, const std::string& profiler_filename, const int64_t workload);

    Variable* operator()(Variable* input_ids,
                         Variable* token_type_ids,
                         Variable* position_ids,
                         Variable* word_embeddings,
                         Variable* token_type_embeddings,
                         Variable* position_embeddings,
                         Variable* gamma,
                         Variable* beta);

    ~EmbeddingAddAddLayerNormOp() = default;

    void ForwardImpl();
};

}  // namespace lightinfer