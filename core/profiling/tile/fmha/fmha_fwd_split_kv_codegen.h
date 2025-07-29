#pragma once

#include <array>
#include <string>

#include "core/profiling/tile/fmha/fmha_fwd_codegen.h"
#include "core/profiling/tile/fmha/fmha_library.h"
#include "core/utils/dtype.h"

namespace flashck {

/**
 * @class FmhaFwdTileDesc
 * @brief Describes the tiling configuration for FMHA operations
 *
 * This class defines how the FMHA computation is divided across thread blocks
 * and how attention computation is tiled across different dimensions.
 * It specifies the work distribution strategy for optimal GPU performance.
 */
class FmhaFwdSplitKVTileDesc {
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
 * @class FmhaFwdSplitKVCodeGen
 * @brief Code generator for Forward FMHA SplitKV operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for Forward Multi-Head Attention SplitKV operations.
 * SplitKV is used for handling very long sequences by splitting key-value computation
 * across multiple blocks and then combining the results.
 */
class FmhaFwdSplitKVCodeGen {
public:
    /**
     * @brief Default constructor with sensible defaults
     */
    FmhaFwdSplitKVCodeGen() = default;

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

    // ====================== Tiling Configuration ======================

    FmhaFwdSplitKVTileDesc tile_desc_;  ///< Tile configuration for this FMHA SplitKV operation

    // ====================== SplitKV Specific Configuration ======================

    bool has_uneven_splits_                  = false;  ///< Enable handling of uneven splits across sequence length
    bool is_store_lse_                       = false;  ///< Enable storing log-sum-exp values for numerical stability
    bool is_merge_num_head_groups_seq_len_q_ = false;  ///< Enable merging head groups with sequence length for Q

    // ====================== Padding Configuration ======================

    bool is_pad_q_seq_len_    = false;  ///< Enable padding for query sequence length
    bool is_pad_kv_seq_len_   = false;  ///< Enable padding for key-value sequence length
    bool is_pad_qk_head_dim_  = false;  ///< Enable padding for query-key head dimension
    bool is_pad_v_head_dim_   = false;  ///< Enable padding for value head dimension
    bool is_pad_qkv_head_dim_ = false;  ///< Enable padding for unified QKV head dimension

    // ====================== Performance Configuration ======================

    int block_per_cu_ = -1;  ///< Override occupancy if not -1 (blocks per compute unit)

    // ====================== Pipeline Configuration ======================

    BlockFmhaPipelineEnum pipeline_ = BlockFmhaPipelineEnum::QRKSVS;  ///< FMHA pipeline implementation variant
};

}  // namespace flashck