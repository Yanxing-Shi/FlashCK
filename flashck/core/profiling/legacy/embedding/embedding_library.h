#pragma once

namespace flashck {

enum class EmbeddingOperationKind {
    SparseEmbedding          = 0,
    SparseEmbeddingLayerNorm = 1,
};

template<EmbeddingOperationKind Kind>
struct EmbeddingOperationTraits;

template<>
struct EmbeddingOperationTraits<EmbeddingOperationKind::SparseEmbedding> {
    static constexpr const char* name = "SparseEmbedding";
};

template<>
struct EmbeddingOperationTraits<EmbeddingOperationKind::SparseEmbeddingLayerNorm> {
    static constexpr const char* name = "SparseEmbeddingLayerNorm";
};

}  // namespace flashck