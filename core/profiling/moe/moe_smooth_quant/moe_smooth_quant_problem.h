#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/moe/moe_library.h"
#include "core/utils/common.h"
namespace flashck {

class MoeSmoothQuantProblem: public ProblemBase<MoeSmoothQuantProblem> {
public:

    std::string GetTypeImpl()
    {
        return "MoeSmoothQuantProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"type\":\"" << GetTypeImpl() << "\",";
        oss << "\"input_tokens\":" << input_tokens_ << ",";
        oss << "\"hidden_size\":" << hidden_size_ << ",";
        oss << "\"num_experts\":" << num_experts_ << ",";
        oss << "\"top_k\":" << topk_ << ",";
        oss << "\"input_stride\":" << input_stride_ << ",";
        oss << "\"output_stride\":" << output_stride_ << ",";
        oss << "\"input_dtype\":\"" << DataTypeToString(input_dtype_) << "\",";
        oss << "\"smooth_scale_dtype\":\"" << DataTypeToString(smooth_scale_dtype_) << "\",";
        oss << "\"output_scale_dtype\":\"" << DataTypeToString(output_scale_dtype_) << "\",";
        oss << "\"output_dtype\":\"" << DataTypeToString(output_dtype_) << "\",";
        oss << "\"acc_dtype\":\"" << DataTypeToString(acc_dtype_) << "\"";
        oss << "}";
        return oss.str();
    }

    std::string GetNameImpl() 
    {
        return Sprintf("{input_dtype}_{smooth_scale_dtype}_{output_scale_dtype}_{output_dtype}",
                       fmt::arg("input_dtype", DataTypeToString(input_dtype_)),
                       fmt::arg("smooth_scale_dtype", DataTypeToString(smooth_scale_dtype_)),
                       fmt::arg("output_scale_dtype", DataTypeToString(output_scale_dtype_)),
                       fmt::arg("output_dtype", DataTypeToString(output_dtype_)));
    }
    
    /// @brief Number of input tokens (batch size * sequence length)
    int64_t input_tokens_;

    /// @brief Input feature dimension (hidden size)
    int64_t hidden_size_;

    // ====================== Expert Configuration ======================
    
    /// @brief Total number of experts in the MoE model
    int64_t num_experts_;
    
    /// @brief Number of top experts selected per token
    int64_t topk_;

    int64_t input_stride_;
    int64_t output_stride_;
    
    // ====================== Data Type Configuration ======================

    DataType input_dtype_;
    DataType smooth_scale_dtype_;
    DataType output_scale_dtype_;
    DataType output_dtype_;
    DataType acc_dtype_ = DataType::FLOAT32;
    
};

}  // namespace flashck