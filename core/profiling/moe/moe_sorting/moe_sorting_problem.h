#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/moe/moe_library.h"
#include "core/utils/common.h"
namespace flashck {

class MoeSortingProblem: public ProblemBase<MoeSortingProblem> {
public:

    std::string GetTypeImpl()
    {
        return "MoeSortingProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"type\":\"" << GetTypeImpl() << "\",";
        oss << "\"input_tokens\":" << input_tokens_ << ",";
        oss << "\"hidden_size\":" << hidden_size_ << ",";
        oss << "\"intermediate_size\":" << intermediate_size_ << ",";
        oss << "\"num_experts\":" << num_experts_ << ",";
        oss << "\"topk\":" << topk_ << ",";
        oss << "\"index_dtype\":\"" << DataTypeToString(index_dtype_) << "\",";
        oss << "\"weight_dtype\":\"" << DataTypeToString(weight_dtype_) << "\"";
        oss << "}";
        return oss.str();
    }

    std::string GetNameImpl() 
    {
        return Sprintf("{index_dtype}_{weight_dtype}",
                       fmt::arg("index_dtype", DataTypeToString(index_dtype_)),
                       fmt::arg("weight_dtype", DataTypeToString(weight_dtype_)));
    }

    
    /// @brief Number of input tokens (batch size * sequence length)
    int64_t input_tokens_;

    /// @brief Input feature dimension (hidden size)
    int64_t hidden_size_;
    
    /// @brief Intermediate dimension size (typically 4x or 8x hidden size)
    int64_t intermediate_size_;

    // ====================== Expert Configuration ======================
    
    /// @brief Total number of experts in the MoE model
    int64_t num_experts_;
    
    /// @brief Number of top experts selected per token
    int64_t topk_;
    
    // ====================== Data Type Configuration ======================

    DataType index_dtype_;
    DataType weight_dtype_;

};

}  // namespace flashck