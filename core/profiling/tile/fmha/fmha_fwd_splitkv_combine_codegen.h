#pragma once

#include <string>

#include "core/profiling/tile/fmha/fmha_library.h"
#include "core/utils/dtype.h"

namespace flashck {

/**
 * @class FmhaSplitKVCombineTileDesc
 * @brief Describes the tiling configuration for FMHA SplitKV Combine operations
 *
 * This class defines how the FMHA SplitKV Combine computation is divided across thread blocks.
 * The combine operation aggregates partial attention results from multiple SplitKV blocks
 * to produce the final attention output for very long sequences.
 */
class FmhaSplitKVCombineTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName() const;

    // ====================== Tile Configuration ======================

    int64_t bm0_;  ///< Tile size along query sequence length dimension
    int64_t bn1_;  ///< Tile size along value head dimension
};

/**
 * @class FmhaFwdSplitKVCombineCodeGen
 * @brief Code generator for Forward FMHA SplitKV Combine operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for Forward Multi-Head Attention SplitKV Combine operations.
 * The combine operation is the second stage of SplitKV processing, which aggregates
 * partial attention results computed across multiple blocks for very long sequences.
 */
class FmhaFwdSplitKVCombineCodeGen {
public:
    /**
     * @brief Default constructor with sensible defaults
     */
    FmhaFwdSplitKVCombineCodeGen() = default;

    /**
     * @brief Generate pipeline configuration name
     * @return String identifier for pipeline configuration
     */
    std::string GetPipelineConfigName() const;

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

    FmhaKind kind_ = FmhaKind::FwdSplitKVCombine;  ///< Type of FMHA operation (always FwdSplitKVCombine)

    // ====================== Data Type Configuration ======================

    DataType dtype_ = DataType::FLOAT16;  ///< Primary data type for attention tensors

    // ====================== Attention Configuration ======================

    FmhaMode mode_ = FmhaMode::Batch;  ///< Batch or Group mode for attention computation

    // ====================== Tiling Configuration ======================

    FmhaSplitKVCombineTileDesc tile_desc_;  ///< Tile configuration for this FMHA SplitKV Combine operation

    // ====================== SplitKV Combine Specific Configuration ======================

    int64_t hdim_           = 64;  ///< Head dimension size for attention computation
    int64_t log_max_splits_ = 8;   ///< Log2 of maximum number of splits allowed

    // ====================== Padding Configuration ======================

    bool is_pad_q_seq_len_  = false;  ///< Enable padding for query sequence length
    bool is_pad_v_head_dim_ = false;  ///< Enable padding for value head dimension

    // ====================== Performance Configuration ======================

    int block_per_cu_ = -1;  ///< Override occupancy if not -1 (blocks per compute unit)

    // ====================== Quantization Configuration ======================

    bool is_static_quant_ = false;  ///< Enable static quantization
};

}  // namespace flashck