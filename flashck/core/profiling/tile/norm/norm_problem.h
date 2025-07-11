#pragma once

#include <sstream>

#include "flashck/core/profiling/problem_base.h"
#include "flashck/core/profiling/tile/norm/norm_library.h"

namespace flashck {

class NormProblem: public ProblemBase<NormProblem> {
public:
    std::string GetTypeImpl()
    {
        return "NormProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{"
            << "\"x_dtype\": \"" << DataTypeToString(x_dtype_) << "\", "
            << "\"y_dtype\": \"" << DataTypeToString(y_dtype_) << "\", "
            << "\"smooth_scale_dtype\": \"" << DataTypeToString(smooth_scale_dtype_) << "\", "
            << "\"y_scale_dtype\": \"" << DataTypeToString(y_scale_dtype_) << "\", "
            << "\"m\": " << m_ << ", "
            << "\"n\": " << n_ << ", "
            << "\"kind\": \"" << norm_map.at(kind_).name << "\", "
            << "\"is_add_bias\": \"" << norm_bias_map.at(is_add_bias_).name << "\", "
            << "\"fused_add\": \"" << fused_add_map.at(fused_add_).name << "\", "
            << "\"fused_quant\": \"" << fused_quant_map.at(fused_quant_).name << "\""
            << "}";
        return oss.str();
    }

    DataType x_dtype_;
    DataType y_dtype_;
    DataType smooth_scale_dtype_;
    DataType y_scale_dtype_;

    int64_t m_;
    int64_t n_;

    NormKind kind_;

    NormBiasEnum is_add_bias_;

    FusedAddEnum   fused_add_;
    FusedQuantEnum fused_quant_;
};

}  // namespace flashck