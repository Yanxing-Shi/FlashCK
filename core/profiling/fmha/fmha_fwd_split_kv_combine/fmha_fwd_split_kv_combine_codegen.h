#pragma once

#include <string>

#include "core/profiling/fmha/fmha_library.h"
#include "core/profiling/fmha/fmha_problem.h"

namespace flashck {

/**
 * @class FmhaFwdSplitKVCombineTileDesc
 * @brief Describes the tiling configuration for FMHA SplitKV Combine operations
 *
 * This class defines how the FMHA SplitKV Combine computation is divided across thread blocks.
 * The combine operation aggregates partial attention results from multiple SplitKV blocks
 * to produce the final attention output for very long sequences.
 */
class FmhaFwdSplitKVCombineTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName();

    // ====================== Tile Configuration ======================

    int64_t n1_block_;  ///< Tile size along value head dimension
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
    std::string GetPipelineConfigName();

    /**
     * @brief Generate a unique instance name for this configuration
     * @return String identifier combining operation type and parameters
     */
    std::string GetInstanceName();

    /**
     * @brief Generate the complete kernel code for this configuration
     * @return String containing the generated GPU kernel code
     */
    std::string Emit();

    FmhaProblem problem_;

    int64_t log_max_splits_;

    FmhaFwdSplitKVCombineTileDesc tile_desc_;

    // ====================== Padding Configuration ======================

    bool is_pad_q_seq_len_  = false;  ///< Enable padding for query sequence length
    bool is_pad_v_head_dim_ = false;  ///< Enable padding for value head dimension

    // ====================== Performance Configuration ======================

    int min_block_per_cu_ = -1;  ///< Override occupancy if not -1 (blocks per compute unit)
};

}  // namespace flashck