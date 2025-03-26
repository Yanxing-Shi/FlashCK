#pragma once

#include "lightinfer/core/module/operations/embedding_ops/embedding_common_op.h"

namespace lightinfer {

/*
embedding operation
*/

template<typename T>
class EmbeddingOp: public EmbeddingCommonOp<T, EmbeddingOp<T>> {
public:
    EmbeddingOp(int64_t     num_embeddings_,
                int64_t     embedding_dims,
                float       epsilon = 0.f,
                std::string op_name = "embedding");

    EmbeddingQueryEntry GetEmbeddingQueryEntry(const std::string& workload);

    EmbeddingProblem GetEmbeddingProblem(const int64_t num_indices);

    std::vector<std::string>
    GenOpProfileCmd(const std::string& profiler_prefix, const std::string& profiler_filename, const int64_t workload);

    Variable* operator()(Variable* x, Variable* weight, Variable* gamma, Variable* beta);

    ~EmbeddingOp() = default;

    void ForwardImpl();
};

}  // namespace lightinfer