#pragma once

#include <array>
#include <string>
#include <vector>

#include "core/profiling/attention/fmha_library.h"
#include "core/profiling/attention/fmha_fwd_batch_prefill/fmha_fwd_batch_prefill_problem.h"

namespace flashck {

/**
 * @class FmhaFwdBatchPrefillTileDesc
 * @brief Describes the hierarchical tiling configuration for FMHA batch prefill operations
 *
 * This class encapsulates the complete tiling strategy for FMHA operations including:
 * - Block-level tiling for Q-K and Attention-V GEMM operations
 * - Warp-level distribution within each block
 * - Warp-tile level memory access patterns
 */
class FmhaFwdBatchPrefillTileDesc {
public:
    /**
     * @brief Generate a unique identifier for this tile configuration
     * @return String identifier encoding all tile parameters
     */
    std::string GetInstanceName();

    /**
     * @brief Generate C++ template parameters for kernel instantiation
     * @return String representation for TileFmhaShape template
     */
    std::string Emit();

    // ====================== Q-K GEMM Block Tiling ======================
    /// Block size for query sequence dimension (M0)
    int64_t m0_block_;
    /// Block size for key sequence dimension (N0) 
    int64_t n0_block_;
    /// Block size for head dimension in Q-K GEMM (K0)
    int64_t k0_block_;
    /// Maximum block size for head dimension (for dynamic sizing)
    int64_t k0_max_block_;

    // ====================== Attention-V GEMM Block Tiling ======================
    /// Block size for value head dimension (N1)
    int64_t n1_block_;
    /// Block size for key sequence in Attention-V GEMM (K1)
    int64_t k1_block_;

    // ====================== Warp Distribution ======================
    /// Number of warps along query sequence dimension (M0)
    int64_t m0_warp_;
    /// Number of warps along key sequence dimension (N0)
    int64_t n0_warp_;
    /// Number of warps along head dimension in Q-K GEMM (K0)
    int64_t k0_warp_;
    
    /// Number of warps along query sequence in Attention-V (M1)
    int64_t m1_warp_;
    /// Number of warps along value head dimension (N1)
    int64_t n1_warp_;
    /// Number of warps along key sequence in Attention-V (K1)
    int64_t k1_warp_;

    // ====================== Warp-Level Tile Sizes ======================
    /// Warp tile size for query sequence dimension (M0)
    int64_t m0_warp_tile_;
    /// Warp tile size for key sequence dimension (N0)
    int64_t n0_warp_tile_;
    /// Warp tile size for head dimension in Q-K GEMM (K0)
    int64_t k0_warp_tile_;
    /// Warp tile size for query sequence in Attention-V (M1)
    int64_t m1_warp_tile_;
    /// Warp tile size for value head dimension (N1)
    int64_t n1_warp_tile_;
    /// Warp tile size for key sequence in Attention-V (K1)
    int64_t k1_warp_tile_;
};

/**
 * @class FmhaFwdBatchPrefillCodeGen
 * @brief Complete code generator for FMHA batch prefill operations
 *
 * This class encapsulates all parameters needed to generate efficient FMHA kernels:
 * - Tiling configuration for optimal memory access patterns
 * - Padding support for arbitrary input sizes
 * - Performance tuning parameters (occupancy control)
 * - Pipeline implementation selection
 */
class FmhaFwdBatchPrefillCodeGen {
public:
    /**
     * @brief Default constructor with sensible defaults
     */
    FmhaFwdBatchPrefillCodeGen() = default;
    
    /**
     * @brief Generate unique instance name combining all configuration parameters
     * @return Comprehensive string identifier for this kernel configuration
     */
    std::string GetInstanceName();

    /**
     * @brief Generate complete GPU kernel code for this configuration
     * @return String containing the generated CUDA/HIP kernel implementation
     */
    std::string Emit();

    FmhaFwdBatchPrefillProblem problem_; 

    // ====================== Tiling Configuration ======================
    FmhaFwdBatchPrefillTileDesc tile_desc_;

    // ====================== Trait Configuration ======================
    /// Enable padding for query sequence length dimension
    bool is_pad_q_seq_len_    = false;
    /// Enable padding for key-value sequence length dimension  
    bool is_pad_kv_seq_len_   = false;
    /// Enable padding for query-key head dimension
    bool is_pad_qk_head_dim_  = false;
    /// Enable padding for value head dimension
    bool is_pad_v_head_dim_   = false;

    bool is_skip_min_q_seq_len_;

    // ====================== Strategy Configuration ======================
    BlockFmhaPipelineEnum pipeline_ = BlockFmhaPipelineEnum::QRKSVS;

    // ====================== Launch Configuration ======================
    int64_t max_thread_per_block_;
    int64_t min_block_per_cu_ = -1;
};

}  // namespace flashck