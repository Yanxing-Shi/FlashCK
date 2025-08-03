#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/norm/norm_library.h"

#include "core/utils/common.h"

namespace flashck {

/**
 * @class LayerNormProblem
 * @brief Represents a normalization operation problem configuration
 *
 */
class LayerNormProblem: public ProblemBase<LayerNormProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for norm problems
     */
    std::string GetTypeImpl()
    {
        return "LayerNormProblem";
    }

    /**
     * @brief Serialize the problem configuration to JSON format
     * @return JSON string representation of the problem parameters
     */
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
            << "\"is_add_bias\": \"" << GetNormBiasName(is_add_bias_) << "\", "
            << "\"fused_add\": \"" << GetFusedAddName(fused_add_) << "\", "
            << "\"fused_quant\": \"" << GetFusedQuantName(fused_quant_) << "\""
            << "}";
        return oss.str();
    }

    // ====================== Problem Configuration ======================

    // Data type specifications
    DataType x_dtype_;             ///< Input tensor data type
    DataType y_dtype_;             ///< Output tensor data type
    DataType smooth_scale_dtype_;  ///< Smoothing scale parameter data type
    DataType y_scale_dtype_;       ///< Output scale parameter data type

    // Tensor dimensions
    int64_t m_;  ///< First dimension size (typically batch or sequence length)
    int64_t n_;  ///< Second dimension size (typically feature dimension)

    // Operation configuration flags
    NormBiasEnum is_add_bias_;  ///< Whether to add bias term

    // Fusion options
    FusedAddEnum   fused_add_;    ///< Type of fused addition operation
    FusedQuantEnum fused_quant_;  ///< Type of fused quantization operation
};

}  // namespace flashck