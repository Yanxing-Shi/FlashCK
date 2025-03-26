#pragma once

#include "lightinfer/core/module/layers/gemm_layers/linear_weight.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_weight.h"

namespace lightinfer {

template<typename T>
struct EmbeddingWeight {
    LinearWeight<T> word_embeddings_;
    LinearWeight<T> token_type_embeddings_;
    LinearWeight<T> position_embeddings_;

    LayerNormWeight<T> layer_norm_weight_;
};
}  // namespace lightinfer