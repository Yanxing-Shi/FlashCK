#pragma once

namespace flashck {

class EmbeddingProblem: public ProblemBase<EmbeddingProblem> {
public:
    std::string GetProblemName() const
    {
        return "EmbeddingProblem";
    }

    std::string Serialize() const
    {
        std::ostringstream oss;
        oss << "EmbeddingProblem: "
            << "num_indices=" << num_indices_ << ", vocab_size=" << vocab_size_
            << ", type_vocab_size=" << type_vocab_size_ << ", max_position_embeddings=" << max_position_embeddings_
            << ", embedding_dims=" << embedding_dims_ << ", emb_dtype=" << emb_dtype_
            << ", index_dtype=" << index_dtype_ << ", gamma_dtype=" << gamma_dtype_ << ", beta_dtype=" << beta_dtype_
            << ", acc_dtype=" << acc_dtype_ << ", y_dtype=" << y_dtype_;
        return oss.str();
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
};

}  // namespace flashck