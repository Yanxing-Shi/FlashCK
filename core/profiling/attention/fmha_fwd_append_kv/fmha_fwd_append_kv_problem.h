#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/attention/fmha_library.h"
#include "core/utils/common.h"

namespace flashck {

/**
 * @class FmhaFwdProblem
 * @brief Represents a Fused Multi-Head Attention (FMHA) operation problem configuration
 *
 * This class encapsulates all the parameters needed to define an FMHA operation,
 * including data types, sequence dimensions, attention parameters, and various
 * optimization options like quantization, bias, and RoPE embeddings.
 * It provides serialization capabilities for problem representation and comparison.
 */
class FmhaFwdProblem: public ProblemBase<FmhaFwdProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for FMHA problems
     */
    std::string GetTypeImpl()
    {
        return "FmhaFwdProblem";
    }

    /**
     * @brief Serialize the problem configuration to JSON format
     * @return JSON string representation of the problem parameters
     */
    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"dtype\": \"" << DataTypeToString(dtype_) << "\",";
        oss << "\"batch_size\": " << batch_size_ << ",";
        oss << "\"q_seq_len\": " << q_seq_len_ << ",";
        oss << "\"q_max_seq_len\": " << q_max_seq_len_ << ",";
        oss << "\"kv_seq_len\": " << kv_seq_len_ << ",";
        oss << "\"new_kv_seq_len\": " << new_kv_seq_len_ << ",";
        oss << "\"q_num_heads\": " << q_num_heads_ << ",";
        oss << "\"kv_num_heads\": " << kv_num_heads_ << ",";
        oss << "\"qk_head_dim\": " << qk_head_dim_ << ",";
        oss << "\"v_head_dim\": " << v_head_dim_ << ",";
        oss << "\"paged_block_size\": " << paged_block_size_ << ",";
        oss << "\"use_batch_cache_idx\": " << (use_batch_cache_idx_ ? "true" : "false") << ",";
        oss << "\"rope_type\": \"" << GetRopeShortName(rope_type_) << "\",";
        oss << "\"rotary_dim\": " << rotary_dim_;
        oss << "}";
        return oss.str();
    }

    std::string GetNameImpl(){
        return Sprintf("{dtype}_{paged_kv}_{rope_type}",
                       fmt::arg("dtype", DataTypeToString(dtype_)),
                       fmt::arg("paged_kv", paged_block_size_>0? "TPB" : "FPB"),
                       fmt::arg("rope_type", GetRopeShortName(rope_type_)));
    }


    // ====================== Problem Configuration ======================

    // Data type specification
    DataType dtype_;  ///< Primary data type for Q, K, V tensors

    // Sequence and batch dimensions
    int64_t batch_size_;     ///< Batch size for the operation
    int64_t q_seq_len_;      ///< Query sequence length (average for group mode)
    int64_t q_max_seq_len_;  ///< Maximum query sequence length
    int64_t kv_seq_len_;     ///< Key-Value sequence length
    int64_t new_kv_seq_len_; ///< New Key-Value sequence length

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

}  // namespace flashck