#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/moe/moe_library.h"
#include "core/utils/dtype.h"

namespace flashck {

class MoeProblem: public ProblemBase<MoeProblem> {
public:

    std::string GetTypeImpl()
    {
        return "MoeProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "MoeGemm_"
            << "input_tokens" << input_tokens_ << "_"
            << "hidden_size" << hidden_size_ << "_"
            << "intermediate_size" << intermediate_size_ << "_"
            << "output_hidden_size" << output_hidden_size_ << "_"
            << "num_experts" << num_experts_ << "_"
            << "top_k" << top_k_ << "_"
            << "capacity_factor" << static_cast<int>(capacity_factor_ * 100) << "_"
            << "input_dtype" << DataTypeToString(input_dtype_) << "_"
            << "weight_dtype" << DataTypeToString(weight_dtype_) << "_"
            << "output_dtype" << DataTypeToString(output_dtype_) << "_"
            << "intermediate_dtype" << DataTypeToString(intermediate_dtype_) << "_"
            << "index_dtype" << DataTypeToString(index_dtype_) << "_"
            << "activation" << GetActivationShortName(activation_) << "_"
            << (use_smooth_quant_ ? "squant" : "nosquant") << "_";
        return oss.str();
    }

    // ====================== Stage 0: Token-to-Intermediate Configuration ======================
    
    /// @brief Number of input tokens (batch size * sequence length)
    int64_t input_tokens_;

    /// @brief Input feature dimension (hidden size)
    int64_t hidden_size_;
    
    /// @brief Intermediate dimension size (typically 4x or 8x hidden size)
    int64_t intermediate_size_;

    // ====================== Stage 1: Intermediate-to-Output Configuration ======================
    
    /// @brief Output feature dimension (typically same as input hidden size)
    int64_t output_hidden_size_;

    // ====================== Expert Configuration ======================
    
    /// @brief Total number of experts in the MoE model
    int64_t num_experts_;
    
    /// @brief Number of top experts selected per token
    int64_t top_k_;
    
    /// @brief Capacity factor for load balancing (1.0 = perfect balance)
    double capacity_factor_ = 1.25;

    // ====================== Data Type Configuration ======================
    
    /// @brief Input token data type
    DataType input_dtype_;
    
    /// @brief Expert weight data type
    DataType weight_dtype_;
    
    /// @brief Output data type
    DataType output_dtype_;
    
    /// @brief Intermediate computation data type
    DataType intermediate_dtype_;
    
    /// @brief Expert index data type
    DataType index_dtype_ = DataType::INT32;

    // ====================== Processing Configuration ======================
    
    /// @brief Activation function used between gate and up projections
    ActivationType activation_;
    
    /// @brief Enable smooth quantization for weights
    bool use_smooth_quant_ = false;

};

}  // namespace flashck