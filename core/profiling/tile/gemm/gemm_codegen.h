#pragma once

#include "core/profiling/tile/gemm/gemm_library.h"
#include "core/profiling/tile/gemm/gemm_problem.h"

#include "core/utils/common.h"

namespace flashck {

namespace tile{

/**
 * @class GemmTileDesc
 * @brief Describes the tiling configuration for GEMM operations
 *
 */
class GemmTileDesc {
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

    int64_t m_block_;
    int64_t n_block_;
    int64_t k_block_;

    int64_t m_warp_;
    int64_t n_warp_;
    int64_t k_warp_;

    int64_t m_warp_tile_;
    int64_t n_warp_tile_;
    int64_t k_warp_tile_;

    bool a_permute_ = false;  ///< Whether to permute tensor A
    bool b_permute_ = false;  ///< Whether to permute tensor B

};

/**
 * @class GemmCodeGen
 * @brief Code generator for GEMM operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for GEMM operations. It combines problem
 * specifications with tiling strategies to produce efficient implementations.
 */
class GemmCodeGen {
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
    
    GemmProblem problem_;

    bool is_pad_m_;
    bool is_pad_n_;
    bool is_pad_k_;

    // Tiling strategy
    GemmTileDesc tile_desc_;  ///< Tile configuration for this operation

    // Pipeline 
    PipelineVersionEnum pipeline_version_;
    PipelineSchedulerEnum pipeline_scheduler_;
    EpilogueEnum epilogue_;

    // ====================== Partitioning Parameters ======================
    int64_t min_block_per_cu_; 
    int64_t num_wave_groups_;
    int64_t tile_partitioner_group_num_;
    int64_t tile_partitioner_m01_;

};

} // namespace tile
}  // namespace flashck