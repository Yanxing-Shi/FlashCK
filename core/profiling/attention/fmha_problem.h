#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/attention/fmha_library.h"
#include "core/utils/common.h"

namespace flashck {

/**
 * @class FmhaProblem
 * @brief Represents a Fused Multi-Head Attention (FMHA) operation problem configuration
 *
 * This class encapsulates all the parameters needed to define an FMHA operation,
 * including data types, sequence dimensions, attention parameters, and various
 * optimization options like quantization, bias, and RoPE embeddings.
 * It provides serialization capabilities for problem representation and comparison.
 */
class FmhaProblem: public ProblemBase<FmhaProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for FMHA problems
     */
    std::string GetTypeImpl()
    {
        return "FmhaProblem";
    }

    /**
     * @brief Serialize the problem configuration to JSON format
     * @return JSON string representation of the problem parameters
     */
    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{"
            << "\"mode\": \"" << GetFmhaModeName(mode_) << "\", "
            << "\"kind\": \"" << GetFmhaKindName(kind_) << "\", "
            << "\"dtype\": \"" << DataTypeToString(dtype_) << "\", "
            << "\"mask_type\": \"" << GetAttentionMaskName(mask_type_) << "\", "
            << "\"window_size\": [" << window_size_[0] << ", " << window_size_[1] << "], "
            << "\"bias_enum\": \"" << GetBiasName(bias_enum_) << "\", "
            << "\"is_static_quant\": " << (is_static_quant_ ? "true" : "false") << ", "
            << "\"batch_size\": " << batch_size_ << ", "
            << "\"q_seq_len\": " << q_seq_len_ << ", "
            << "\"q_max_seq_len\": " << q_max_seq_len_ << ", "
            << "\"kv_seq_len\": " << kv_seq_len_ << ", "
            << "\"q_num_heads\": " << q_num_heads_ << ", "
            << "\"kv_num_heads\": " << kv_num_heads_ << ", "
            << "\"qk_head_dim\": " << qk_head_dim_ << ", "
            << "\"v_head_dim\": " << v_head_dim_ << ", "
            << "\"paged_block_size\": " << paged_block_size_ << ", "
            << "\"use_batch_cache_idx\": " << (use_batch_cache_idx_ ? "true" : "false") << ", "
            << "\"rope_type\": \"" << GetRopeName(rope_type_) << "\", "
            << "\"rotary_dim\": " << rotary_dim_ << ", "
            << "\"num_splits\": " << num_splits_ << "}";
        return oss.str();
    }

    // ====================== Problem Configuration ======================

    // Data type specification
    DataType dtype_;  ///< Primary data type for Q, K, V tensors

    // Operation mode and type
    FmhaMode mode_;  ///< Batch or Group mode for attention computation
    FmhaKind kind_;  ///< Type of FMHA operation (Fwd, FwdAppendKV, etc.)

    // Attention configuration
    GenericAttentionMaskEnum mask_type_;    ///< Type of attention mask applied
    std::array<int64_t, 2>   window_size_;  ///< Sliding window size [left, right]
    BiasEnum                 bias_enum_;    ///< Type of bias applied to attention
    int64_t bias_rank_info_; ///< Rank information for bias tensor

    // Quantization configuration
    bool is_static_quant_;  ///< Whether to use static quantization

    // Sequence and batch dimensions
    int64_t batch_size_;     ///< Batch size for the operation
    int64_t q_seq_len_;      ///< Query sequence length (average for group mode)
    int64_t q_max_seq_len_;  ///< Maximum query sequence length
    int64_t kv_seq_len_;     ///< Key-Value sequence length

    // Head configuration
    int64_t q_num_heads_;   ///< Number of query heads
    int64_t kv_num_heads_;  ///< Number of key-value heads (for GQA/MQA)

    // Dimension configuration
    int64_t qk_head_dim_;  ///< Query-Key head dimension
    int64_t v_head_dim_;   ///< Value head dimension

    // Memory and optimization configuration
    int64_t paged_block_size_;     ///< Block size for paged attention
    bool    use_batch_cache_idx_;  ///< Enable batch cache index optimization

    // RoPE (Rotary Position Embedding) configuration
    RopeEnum rope_type_;   ///< Type of rotary position embedding
    int64_t  rotary_dim_;  ///< Rotary embedding dimension

    // Split-KV configuration
    int64_t num_splits_;  ///< Number of splits for key/value (for SplitKV variants)

    bool has_logits_soft_cap_;
    bool is_skip_min_q_seqlen_ = false; //  skip min seqlen q while chunked prefill
    bool is_store_lse_ = false;  ///< Enable storing log-sum-exp values for numerical stability
};
}  // namespace flashck