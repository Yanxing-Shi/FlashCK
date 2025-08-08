#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/moe/moe_library.h"
#include "core/utils/common.h"

namespace flashck {

class MoeGemmProblem: public ProblemBase<MoeGemmProblem> {
public:

    std::string GetTypeImpl()
    {
        return "MoeGemmProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"input_tokens\": " << input_tokens_ << ",";
        oss << "\"hidden_size\": " << hidden_size_ << ",";
        oss << "\"intermediate_size\": " << intermediate_size_ << ",";
        oss << "\"num_experts\": " << num_experts_ << ",";
        oss << "\"topk\": " << topk_ << ",";
        oss << "\"input_dtype\": \"" << DataTypeToString(input_dtype_) << "\",";
        oss << "\"gate_weight_dtype\": \"" << DataTypeToString(gate_weight_dtype_) << "\",";
        oss << "\"down_projection_dtype\": \"" << DataTypeToString(down_projection_dtype_) << "\",";
        oss << "\"acc_dtype\": \"" << DataTypeToString(acc_dtype_) << "\",";
        oss << "\"output_dtype\": \"" << DataTypeToString(output_dtype_) << "\",";
        oss << "\"input_scale_dtype\": \"" << DataTypeToString(input_scale_dtype_) << "\",";
        oss << "\"gate_weight_scale_dtype\": \"" << DataTypeToString(gate_weight_scale_dtype_) << "\",";
        oss << "\"down_projection_scale_dtype\": \"" << DataTypeToString(down_projection_scale_dtype_) << "\",";
        oss << "\"output_scale_dtype\": \"" << DataTypeToString(output_scale_dtype_) << "\",";
        oss << "\"topk_weight_dtype\": \"" << DataTypeToString(topk_weight_dtype_) << "\",";
        oss << "\"index_dtype\": \"" << DataTypeToString(index_dtype_) << "\",";
        oss << "\"activation\": \"" << GetActivationShortName(activation_) << "\",";
        oss << "\"is_only_gate\": " << (is_only_gate_ ? "true" : "false") << ",";
        oss << "\"use_smooth_quant\": " << (use_smooth_quant_ ? "true" : "false");
        oss << "}";
        return oss.str();
    }

    std::string GetNameImpl() 
    {
        return Sprintf("{input_dtype}_{gate_weight_dtype}_{down_projection_dtype}_{output_dtype}_{input_scale_dtype}_{gate_weight_scale_dtype}_{down_projection_scale_dtype}"
                        "_{output_scale_dtype}_{topk_weight_dtype}_{index_dtype}_{activation}_{is_only_gate}_{use_smooth_quant}",
                       fmt::arg("input_dtype", DataTypeToString(input_dtype_)),
                       fmt::arg("gate_weight_dtype", DataTypeToString(gate_weight_dtype_)),
                       fmt::arg("down_projection_dtype", DataTypeToString(down_projection_dtype_)),
                       fmt::arg("output_dtype", DataTypeToString(output_dtype_)),
                       fmt::arg("input_scale_dtype", DataTypeToString(input_scale_dtype_)),
                       fmt::arg("gate_weight_scale_dtype", DataTypeToString(gate_weight_scale_dtype_)),
                       fmt::arg("down_projection_scale_dtype", DataTypeToString(down_projection_scale_dtype_)),
                       fmt::arg("output_scale_dtype", DataTypeToString(output_scale_dtype_)),
                       fmt::arg("topk_weight_dtype", DataTypeToString(topk_weight_dtype_)),
                       fmt::arg("index_dtype", DataTypeToString(index_dtype_)),
                       fmt::arg("activation", GetActivationShortName(activation_)),
                       fmt::arg("is_only_gate", is_only_gate_),
                       fmt::arg("use_smooth_quant", use_smooth_quant_));
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

    DataType input_dtype_;
    DataType gate_weight_dtype_;
    DataType down_projection_dtype_;
    DataType acc_dtype_ = DataType::FLOAT32;
    DataType output_dtype_;

    DataType input_scale_dtype_;
    DataType gate_weight_scale_dtype_;
    DataType down_projection_scale_dtype_;
    DataType output_scale_dtype_;

    DataType topk_weight_dtype_;
    DataType index_dtype_ = DataType::INT32;
    
    ActivationType activation_;
    
    bool is_only_gate_ = false;
    bool use_smooth_quant_ = false;
    

};

}  // namespace flashck