#pragma once

#include <cstdint>
#include <string>

#include "flashck/core/profiling/tile/norm/norm_library.h"
#include "flashck/core/utils/common.h"

namespace flashck {

/**
 * @class NormTileDesc
 * @brief Describes the tile configuration for norm operations
 *
 * This class encapsulates the thread block and work distribution parameters
 * for norm kernel tile configurations, including memory access patterns
 * and vectorization settings.
 */
class NormTileDesc {
public:
    NormTileDesc() = default;

    /**
     * @brief Constructor with initialization parameters
     * @param repeat_m Thread repetition along M dimension
     * @param repeat_n Thread repetition along N dimension
     * @param thread_per_block_m Number of threads per block along M
     * @param thread_per_block_n Number of threads per block along N
     * @param vector_n Vector size along N dimension
     */
    NormTileDesc(
        int64_t repeat_m, int64_t repeat_n, int64_t thread_per_block_m, int64_t thread_per_block_n, int64_t vector_n):
        repeat_m_(repeat_m),
        repeat_n_(repeat_n),
        thread_per_block_m_(thread_per_block_m),
        thread_per_block_n_(thread_per_block_n),
        vector_n_(vector_n)
    {
    }

    /**
     * @brief Generates configuration name string for this tile descriptor
     * @return String representation of the tile configuration
     */
    std::string GetConfigName() const;

    /**
     * @brief Emits the tile descriptor code template
     * @return Generated code string for the tile configuration
     */
    std::string Emit() const;

    /**
     * @brief Generates a human-readable string representation
     * @return String describing the tile configuration
     */
    std::string ToString() const;

    /**
     * @brief Validates if the tile descriptor parameters are valid
     * @return true if valid, false otherwise
     */
    bool IsValid() const;

    /**
     * @brief Calculates the total number of threads in the block
     * @return Total thread count
     */
    int64_t GetTotalThreads() const
    {
        return thread_per_block_m_ * thread_per_block_n_;
    }

    /**
     * @brief Calculates the effective block size along M dimension
     * @return Effective M dimension size
     */
    int64_t GetEffectiveM() const
    {
        return repeat_m_ * thread_per_block_m_;
    }

    /**
     * @brief Calculates the effective block size along N dimension
     * @return Effective N dimension size
     */
    int64_t GetEffectiveN() const
    {
        return repeat_n_ * thread_per_block_n_ * vector_n_;
    }

    // Public member variables for direct access
    int64_t repeat_m_           = 1;  ///< Each thread repeat along M dimension
    int64_t repeat_n_           = 1;  ///< Each thread repeat along N dimension
    int64_t thread_per_block_m_ = 1;  ///< Number of threads along M dimension
    int64_t thread_per_block_n_ = 1;  ///< Number of threads along N dimension
    int64_t vector_n_           = 1;  ///< Vector size along N dimension

private:
    /**
     * @brief Calculates warp distribution parameters
     * @return Pair of (warps_m, warps_n)
     */
    std::pair<int64_t, int64_t> CalculateWarpDistribution() const;
};

/**
 * @class NormCodeGen
 * @brief Generates code for norm operations with specified configuration
 *
 * This class manages the code generation for normalization operations,
 * including data type configuration, fusion options, and tile descriptor
 * integration.
 */
class NormCodeGen {
public:
    NormCodeGen() = default;

    /**
     * @brief Constructor with full configuration
     * @param kind Normalization kind (LayerNorm, RMSNorm, etc.)
     * @param x_dtype Input data type
     * @param y_dtype Output data type
     * @param smooth_scale_dtype Smooth scale data type
     * @param y_scale_dtype Output scale data type
     * @param tile_desc Tile descriptor configuration
     * @param is_add_bias Bias addition mode
     * @param fused_add Fused addition mode
     * @param fused_quant Fused quantization mode
     */
    NormCodeGen(NormKind            kind,
                DataType            x_dtype,
                DataType            y_dtype,
                DataType            smooth_scale_dtype,
                DataType            y_scale_dtype,
                const NormTileDesc& tile_desc,
                NormBiasEnum        is_add_bias,
                FusedAddEnum        fused_add,
                FusedQuantEnum      fused_quant):
        kind_(kind),
        x_dtype_(x_dtype),
        y_dtype_(y_dtype),
        smooth_scale_dtype_(smooth_scale_dtype),
        y_scale_dtype_(y_scale_dtype),
        tile_desc_(tile_desc),
        is_add_bias_(is_add_bias),
        fused_add_(fused_add),
        fused_quant_(fused_quant)
    {
    }

    /**
     * @brief Generates unique configuration name for this norm operation
     * @return String identifier for the configuration
     */
    std::string GetConfigName() const;

    /**
     * @brief Emits the complete code for the norm operation
     * @return Generated code string
     */
    std::string Emit() const;

    /**
     * @brief Validates the current configuration
     * @return true if configuration is valid, false otherwise
     */
    bool IsValid() const;

    /**
     * @brief Generates a human-readable description of the configuration
     * @return String describing the norm configuration
     */
    std::string ToString() const;

    /**
     * @brief Checks if the configuration uses quantization
     * @return true if quantization is enabled
     */
    bool IsQuantizationEnabled() const
    {
        return fused_quant_ != FusedQuantEnum::NO_SWEEP;
    }

    /**
     * @brief Checks if the configuration uses bias addition
     * @return true if bias addition is enabled
     */
    bool IsBiasEnabled() const
    {
        return is_add_bias_ != NormBiasEnum::NO_BIAS;
    }

    /**
     * @brief Checks if the configuration uses fused addition
     * @return true if fused addition is enabled
     */
    bool IsFusedAddEnabled() const
    {
        return fused_add_ != FusedAddEnum::NO_ADD;
    }

    // Public member variables for configuration
    NormKind kind_ = NormKind::LayerNorm;  ///< Normalization kind

    DataType x_dtype_            = DataType::FLOAT32;  ///< Input data type
    DataType y_dtype_            = DataType::FLOAT32;  ///< Output data type
    DataType smooth_scale_dtype_ = DataType::FLOAT32;  ///< Smooth scale data type
    DataType y_scale_dtype_      = DataType::FLOAT32;  ///< Output scale data type

    NormTileDesc tile_desc_;  ///< Tile descriptor

    NormBiasEnum   is_add_bias_ = NormBiasEnum::NO_BIAS;     ///< Bias addition mode
    FusedAddEnum   fused_add_   = FusedAddEnum::NO_ADD;      ///< Fused addition mode
    FusedQuantEnum fused_quant_ = FusedQuantEnum::NO_SWEEP;  ///< Fused quantization mode

private:
    /**
     * @brief Generates template value map for code generation
     * @return Map of template variables and their values
     */
    jinja2::ValuesMap GenerateValueMap() const;

    /**
     * @brief Gets the appropriate template source based on configuration
     * @return Template source string
     */
    std::string GetTemplateSource() const;

    /**
     * @brief Validates data type compatibility
     * @return true if data types are compatible
     */
    bool ValidateDataTypes() const;

    /**
     * @brief Validates fusion mode compatibility
     * @return true if fusion modes are compatible
     */
    bool ValidateFusionModes() const;

    static int instance_counter_;  ///< Counter for unique instance IDs
};

}  // namespace flashck