#pragma once

#include <string>

#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/graph/node.h"

#include "lightinfer/core/module/operations/embedding_ops/embedding_add_add_layer_norm_op.h"
#include "lightinfer/core/module/operations/embedding_ops/embedding_op.h"

namespace lightinfer {

template<typename T>
class EmbeddingLayer: public Layer {
public:
    // bert emebdding layer fused with layernorm
    EmbeddingLayer(int64_t vocab_size,
                   int64_t type_vocab_size,
                   int64_t max_position_embeddings,
                   int64_t embedding_dims,
                   float   epsilon = 1e-12);
    Variable* operator()(Variable* input_ids, Variable* token_type_ids, Variable* position_ids);
    void      LoadParam(const T* word_embeddings_ptr,
                        const T* token_type_embeddings_ptr,
                        const T* position_embeddings_ptr,
                        const T* gamma_ptr,
                        const T* beta_ptr);

    // general
    EmbeddingLayer(int64_t num_embeddings, int64_t embedding_dims, float epsilon = 0.f);
    Variable* operator()(Variable* input_ids);
    void      LoadParam(const T* weight_ptr, const T* gamma_ptr, const T* beta_ptr);

    ~EmbeddingLayer() = default;

private:
    int64_t num_embeddings_;
    int64_t vocab_size_;
    int64_t type_vocab_size_;
    int64_t max_position_embeddings_;

    int64_t embedding_dims_;

    std::unique_ptr<EmbeddingOp<T>>                embedding_op_;
    std::unique_ptr<EmbeddingAddAddLayerNormOp<T>> embedding_add_add_layer_norm_op_;

    std::unique_ptr<Variable> weight_var_;
    std::unique_ptr<Variable> word_embeddings_var_;
    std::unique_ptr<Variable> token_type_embeddings_var_;
    std::unique_ptr<Variable> position_embeddings_var_;
    std::unique_ptr<Variable> gamma_var_;
    std::unique_ptr<Variable> beta_var_;
};

}  // namespace lightinfer