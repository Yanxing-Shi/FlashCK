#pragma once

#include "flashck/core/profiling/tile/norm/norm_library.h"

#include "flashck/core/utils/common.h"

namespace flashck {

/**
 * @class NormTileDesc
 * @brief Describes the tiling configuration for normalization operations
 *
 * This class defines how the normalization computation is divided across
 * thread blocks and individual threads. It specifies the work distribution
 * and vectorization strategy for optimal GPU performance.
 */
class NormTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate code template parameters for this tile
     * @return String representation for code generation
     */
    std::string Emit() const;

    // ====================== Tile Configuration Parameters ======================

    int64_t repeat_m_;            ///< Number of M-dimension repeats per thread
    int64_t repeat_n_;            ///< Number of N-dimension repeats per thread
    int64_t thread_per_block_m_;  ///< Number of threads along M dimension in a block
    int64_t thread_per_block_n_;  ///< Number of threads along N dimension in a block
    int64_t vector_n_;            ///< Vector size along N dimension for memory coalescing
};

/**
 * @class NormCodeGen
 * @brief Code generator for normalization operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for normalization operations. It combines problem
 * specifications with tiling strategies to produce efficient implementations.
 */
class NormCodeGen {
public:
    /**
     * @brief Generate a unique instance name for this configuration
     * @return String identifier combining operation type and parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate the complete kernel code for this configuration
     * @return String containing the generated GPU kernel code
     */
    std::string Emit() const;

    // ====================== Operation Configuration ======================

    NormKind kind_;  ///< Type of normalization operation (LayerNorm, RMSNorm, etc.)

    // Data type specifications
    DataType x_dtype_;             ///< Input tensor data type
    DataType y_dtype_;             ///< Output tensor data type
    DataType smooth_scale_dtype_;  ///< Smoothing scale parameter data type
    DataType y_scale_dtype_;       ///< Output scale parameter data type

    // Tiling strategy
    NormTileDesc tile_desc_;  ///< Tile configuration for this operation

    // Operation flags and fusion options
    NormBiasEnum   is_add_bias_;  ///< Whether to include bias addition
    FusedAddEnum   fused_add_;    ///< Type of fused addition operation
    FusedQuantEnum fused_quant_;  ///< Type of fused quantization operation
};

}  // namespace flashck