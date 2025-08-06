#pragma once

#include <array>
#include <string>
#include <vector>

#include "core/profiling/attention/fmha_library.h"
#include "core/profiling/attention/fmha_problem.h"

namespace flashck {

/**
 * @class FmhaFwdTileDesc
 * @brief Describes the tiling configuration for FMHA operations
 *
 * This class defines how the FMHA computation is divided across thread blocks
 * and how attention computation is tiled across different dimensions.
 * It specifies the work distribution strategy for optimal GPU performance.
 */
class FmhaFwdPagedKVPrefillTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName();

    /**
     * @brief Generate code template parameters for this tile
     * @return String representation for code generation
     */
    std::string Emit();

    // ====================== Q-K GEMM Tile Configuration ======================
    int64_t m0_block_;
    int64_t n0_block_;
    int64_t k0_block_;
    int64_t k0_max_block_;

    int64_t n1_block_;
    int64_t k1_block_;

    // ====================== Warp Distribution ======================
    int64_t m0_warp_;
    int64_t n0_warp_;
    int64_t k0_warp_;

    int64_t m1_warp_;
    int64_t n1_warp_;
    int64_t k1_warp_;

    // ====================== Warp-Level Tile Sizes ======================
    int64_t m0_warp_tile_;
    int64_t n0_warp_tile_;
    int64_t k0_warp_tile_;
    int64_t m1_warp_tile_;
    int64_t n1_warp_tile_;
    int64_t k1_warp_tile_;

};

/**
 * @class FmhaFwdCodeGen
 * @brief Code generator for Forward FMHA operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for Forward Multi-Head Attention operations. It combines
 * problem specifications with tiling strategies and attention-specific configurations.
 */
class FmhaFwdPagedKVPrefillCodeGen {
public:
    /**
     * @brief Default constructor with sensible defaults
     */
    FmhaFwdPagedKVPrefillCodeGen() = default;

    /**
     * @brief Generate padding configuration name
     * @return String identifier for padding configuration
     */
    std::string GetPaddingConfigName();

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

    FmhaFwdPagedKVPrefillProblem problem_; 

    // ====================== Tiling Configuration ======================

    FmhaFwdPagedKVPrefillTileDesc tile_desc_;  ///< Tile configuration for this FMHA operation

    // ====================== Padding Configuration ======================

    bool is_pad_q_seq_len_    = false;  ///< Enable padding for query sequence length
    bool is_pad_kv_seq_len_   = false;  ///< Enable padding for key-value sequence length
    bool is_pad_qk_head_dim_  = false;  ///< Enable padding for query-key head dimension
    bool is_pad_v_head_dim_   = false;  ///< Enable padding for value head dimension

    // ====================== Performance Configuration ======================

    int min_block_per_cu_ = -1;  ///< Override occupancy if not -1 (blocks per compute unit)
    
    // ====================== Pipeline Configuration ======================

    BlockFmhaPipelineEnum pipeline_ = BlockFmhaPipelineEnum::QRKSVS;  ///< FMHA pipeline implementation variant

    bool is_skip_min_q_seqlen_;

};

}  // namespace flashck