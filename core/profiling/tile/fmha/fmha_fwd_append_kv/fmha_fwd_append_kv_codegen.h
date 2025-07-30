#pragma once

#include <array>
#include <string>

#include "core/profiling/tile/fmha/fmha_library.h"
#include "core/profiling/tile/fmha/fmha_problem.h"

namespace flashck {

/**
 * @class FmhaFwdAppendKVTileDesc
 * @brief Describes the tiling configuration for FMHA AppendKV operations
 *
 * This class defines how the FMHA AppendKV computation is divided across thread blocks.
 * AppendKV operations handle dynamic key-value cache updates in attention computation.
 */
class FmhaFwdAppendKVTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName() const;

    // ====================== Tile Configuration ======================

    int64_t s_block_;   ///< Tile size along query sequence length
    int64_t sk_block_;  ///< Tile size along key sequence length
    int64_t d_block_;   ///< Tile size along Q-K GEMM unroll dimension
    int64_t dv_block_;  ///< Tile size along K-V GEMM unroll dimension
};

/**
 * @class FmhaFwdAppendKVCodeGen
 * @brief Code generator for Forward FMHA AppendKV operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for Forward Multi-Head Attention AppendKV operations.
 * AppendKV is used for dynamic key-value cache updates during inference.
 */
class FmhaFwdAppendKVCodeGen {
public:
    /**
     * @brief Default constructor with sensible defaults
     */
    FmhaFwdAppendKVCodeGen() = default;

    /**
     * @brief Generate padding configuration name
     * @return String identifier for padding configuration
     */
    std::string GetPadName() const;

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

    FmhaProblem problem_;

    FmhaFwdAppendKVTileDesc tile_desc_;  ///< Tile configuration for this FMHA AppendKV operation

    // ====================== Padding Configuration ======================

    bool is_pad_q_seq_len_    = false;  ///< Enable padding for query sequence length
    bool is_pad_kv_seq_len_   = false;  ///< Enable padding for key-value sequence length
    bool is_pad_qk_head_dim_  = false;  ///< Enable padding for query-key head dimension
    bool is_pad_v_head_dim_   = false;  ///< Enable padding for value head dimension

    // ====================== Performance Configuration ======================

    int min_block_per_cu_ = -1;  ///< Override occupancy if not -1 (blocks per compute unit)
};

}  // namespace flashck