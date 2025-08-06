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
class FmhaFwdSplitKVCombineProblem: public ProblemBase<FmhaFwdSplitKVCombineProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for FMHA problems
     */
    std::string GetTypeImpl()
    {
        return "FmhaFwdSplitKVCombineProblem";
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
        oss << "\"mode\": \"" << GetFmhaModeName(mode_) << "\",";
        oss << "\"is_static_quant\": " << (is_static_quant_ ? "true" : "false") << ",";
        oss << "\"batch_size\": " << batch_size_ << ",";
        oss << "\"q_seq_len\": " << q_seq_len_ << ",";
        oss << "\"q_max_seq_len\": " << q_max_seq_len_ << ",";
        oss << "\"kv_seq_len\": " << kv_seq_len_ << ",";
        oss << "\"q_num_heads\": " << q_num_heads_ << ",";
        oss << "\"kv_num_heads\": " << kv_num_heads_ << ",";
        oss << "\"qk_head_dim\": " << qk_head_dim_ << ",";
        oss << "\"v_head_dim\": " << v_head_dim_;
        oss << "\"num_splits\": " << num_splits_;
        oss << "}";
        return oss.str();
    }

    std::string GetNameImpl(){
        return Sprintf("{dtype}_{mode}_{is_static_quant}",
                       fmt::arg("dtype", DataTypeToString(dtype_)),
                       fmt::arg("mode", GetFmhaModeShortName(mode_)),
                       fmt::arg("is_static_quant", is_static_quant_));
    }

    // ====================== Problem Configuration ======================

    // Data type specification
    DataType dtype_;  ///< Primary data type for Q, K, V tensors

    // Operation mode and type
    FmhaMode mode_;  ///< Batch or Group mode for attention computation

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

    int64_t num_splits_;

};
}  // namespace flashck