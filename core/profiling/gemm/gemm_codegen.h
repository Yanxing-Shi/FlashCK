#pragma once

#include "core/profiling/gemm/gemm_library.h"
#include "core/profiling/gemm/gemm_problem.h"

#include "core/utils/common.h"

namespace flashck {

/**
 * @class GemmTileDesc
 * @brief High-performance GEMM tiling configuration with hierarchical memory optimization
 *
 * This class encapsulates the complete tiling strategy for GEMM operations, implementing a
 * three-level hierarchical approach that maximizes GPU performance through optimized memory
 * access patterns and computational efficiency.
 */
class GemmTileDesc {
public:
    /**
     * @brief Generate comprehensive unique identifier for this tile configuration
     * @return String identifier encoding all tile parameters for profiling and caching
     */
    std::string GetInstanceName();

    /**
     * @brief Generate optimized code template parameters for GPU kernel instantiation
     * @return Template string with tile configuration for code generation
     */
    std::string Emit();

    // ====================== Hierarchical Tiling Parameters ======================

    // Block-level tiling (shared memory allocation and CU work distribution)
    int64_t m_block_;        ///< M-dimension block size (rows of result matrix C)
    int64_t n_block_;        ///< N-dimension block size (columns of result matrix C)
    int64_t k_block_;        ///< K-dimension block size (reduction dimension)

    // Warp-level tiling (SIMD execution configuration)
    int64_t m_warp_;         ///< Number of warps in M-dimension per block
    int64_t n_warp_;         ///< Number of warps in N-dimension per block
    int64_t k_warp_;         ///< Number of warps in K-dimension per block

    // Thread-level tiling (per-thread register allocation)
    int64_t m_warp_tile_;    ///< Elements per thread in M-dimension within warp
    int64_t n_warp_tile_;    ///< Elements per thread in N-dimension within warp
    int64_t k_warp_tile_;    ///< Elements per thread in K-dimension within warp

    // Memory layout optimization flags
    bool a_permute_ = false; ///< Enable tensor A layout permutation for cache efficiency
    bool b_permute_ = false; ///< Enable tensor B layout permutation for cache efficiency
};

/**
 * @class GemmCodeGen
 * @brief Comprehensive GEMM operation code generator with advanced optimization features
 *
 * This class provides complete code generation capabilities for high-performance GEMM
 * operations on GPU architectures. It combines problem specifications, tiling strategies,
 * pipeline configurations, and hardware-specific optimizations to produce efficient kernels.
 *
 */
class GemmCodeGen {
public:
    /**
     * @brief Generate unique instance identifier combining all configuration parameters
     * @return Comprehensive string identifier for profiling, caching, and debugging
     */
    std::string GetInstanceName();

    /**
     * @brief Generate complete optimized GPU kernel code for this GEMM configuration
     * @return String containing full kernel implementation with all optimizations applied
     */
    std::string Emit();

    // ====================== Core Operation Configuration ======================
    
    GemmProblem problem_;    ///< Complete problem specification (dimensions, types, layouts)

    // Padding configuration for non-aligned dimensions
    bool is_pad_m_;          ///< Enable M-dimension padding for alignment
    bool is_pad_n_;          ///< Enable N-dimension padding for alignment
    bool is_pad_k_;          ///< Enable K-dimension padding for alignment

    // Hierarchical tiling strategy
    GemmTileDesc tile_desc_; ///< Complete tile configuration for optimal performance

    // ====================== Pipeline Configuration ======================
    
    PipelineVersionEnum pipeline_version_;    ///< Pipeline version for computation strategy
    PipelineSchedulerEnum pipeline_scheduler_; ///< Work scheduling strategy
    EpilogueEnum pipeline_epilogue_;          ///< Post-computation operations (bias, activation)

    // ====================== Advanced Partitioning Parameters ======================
    
    int64_t min_block_per_cu_;           ///< Minimum blocks per compute unit for occupancy control
    int64_t num_wave_groups_;            ///< Number of wave groups for advanced scheduling
    int64_t tile_partitioner_group_num_; ///< Group number for tile partitioning strategies
    int64_t tile_partitioner_m01_;       ///< M-dimension partitioning parameter for load balancing
};

}  // namespace flashck