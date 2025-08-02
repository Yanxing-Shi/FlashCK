#pragma once

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_problem.h"

#include "core/utils/common.h"

namespace flashck {

/**
 * @class MoeGemmTileDesc
 * @brief High-performance MoE (Mixture of Experts) GEMM tiling configuration with dual-stage optimization
 *
 * This class encapsulates the complete tiling strategy for MoE GEMM operations, implementing a
 * sophisticated dual-stage GEMM approach that maximizes expert routing efficiency and computational
 * throughput in transformer MoE architectures.
 *
 * Architecture Overview:
 * ======================
 * Dual-Stage MoE GEMM Processing:
 *   Stage 0 (Token-to-Intermediate): Input tokens → Intermediate representations
 *   Stage 1 (Intermediate-to-Output): Intermediate representations → Output tokens
 *
 * Each stage uses hierarchical tiling:
 *   Block Level (GPU CU/SM): Controls memory bandwidth and expert parallelism
 *   Warp Level (GPU warp): Manages SIMD execution and expert workload distribution
 *   Thread Level (GPU thread): Optimizes register usage for sparse expert access patterns
 *
 * Expert Routing Optimization:
 *   - Efficient sparse expert selection and routing
 *   - Load balancing across heterogeneous expert workloads
 *   - Minimized communication overhead between routing stages
 *
 * Memory Access Patterns:
 * =======================
 * - Coalesced access for expert weight matrices
 * - Optimized routing table lookup patterns
 * - Efficient intermediate result caching between stages
 * - Hardware-specific alignment for maximum throughput
 *
 * Performance Optimization:
 * =========================
 * - Block sizes tuned for expert sparsity patterns
 * - Warp configurations optimized for routing efficiency
 * - Thread tiles balanced for register pressure in sparse computations
 */
class MoeGemmTileDesc {
public:
    /**
     * @brief Update block tile dimensions for dual-stage MoE GEMM
     * @param block_tile Vector containing: [m0_block, n0_block, k0_block, m1_block, n1_block, k1_block]
     */
    void UpdateBlockTile(std::vector<int64_t> block_tile){
        m0_block_ = block_tile[0];
        n0_block_ = block_tile[1];
        k0_block_ = block_tile[2];
        m1_block_ = block_tile[3];
        n1_block_ = block_tile[4];
        k1_block_ = block_tile[5];
    }

    /**
     * @brief Generate comprehensive unique identifier for this MoE tile configuration
     * @return String identifier encoding all dual-stage tile parameters for profiling and caching
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate optimized code template parameters for MoE GPU kernel instantiation
     * @return Template string with dual-stage tile configuration for code generation
     */
    std::string Emit() const;

    // ====================== Stage 0: Token-to-Intermediate GEMM Tiling ======================
    
    int64_t m0_block_;       ///< M-dimension block size for stage 0 (token dimension)
    int64_t n0_block_;       ///< N-dimension block size for stage 0 (intermediate dimension)
    int64_t k0_block_;       ///< K-dimension block size for stage 0 (input hidden dimension)

    // ====================== Stage 1: Intermediate-to-Output GEMM Tiling ======================
    
    int64_t m1_block_;       ///< M-dimension block size for stage 1 (token dimension)
    int64_t n1_block_;       ///< N-dimension block size for stage 1 (output dimension)
    int64_t k1_block_;       ///< K-dimension block size for stage 1 (intermediate dimension)

    // ====================== Stage 0 Warp Distribution ======================
    
    int64_t m0_warp_;        ///< Number of warps in M-dimension for stage 0
    int64_t n0_warp_;        ///< Number of warps in N-dimension for stage 0
    int64_t k0_warp_;        ///< Number of warps in K-dimension for stage 0

    // ====================== Stage 1 Warp Distribution ======================
    
    int64_t m1_warp_;        ///< Number of warps in M-dimension for stage 1
    int64_t n1_warp_;        ///< Number of warps in N-dimension for stage 1
    int64_t k1_warp_;        ///< Number of warps in K-dimension for stage 1

    // ====================== Stage 0 Thread-Level Tile Sizes ======================
    
    int64_t m0_warp_tile_;   ///< Elements per thread in M-dimension for stage 0
    int64_t n0_warp_tile_;   ///< Elements per thread in N-dimension for stage 0
    int64_t k0_warp_tile_;   ///< Elements per thread in K-dimension for stage 0

    // ====================== Stage 1 Thread-Level Tile Sizes ======================
    
    int64_t m1_warp_tile_;   ///< Elements per thread in M-dimension for stage 1
    int64_t n1_warp_tile_;   ///< Elements per thread in N-dimension for stage 1
    int64_t k1_warp_tile_;   ///< Elements per thread in K-dimension for stage 1
};

/**
 * @class MoeGemmCodeGen
 * @brief Comprehensive MoE GEMM operation code generator with expert routing optimization
 *
 * This class provides complete code generation capabilities for high-performance MoE GEMM
 * operations on GPU architectures. It combines dual-stage GEMM computations with efficient
 * expert routing, load balancing, and memory optimization strategies.
 *
 * Code Generation Features:
 * =========================
 * Expert Routing:
 *   - Sparse expert selection based on routing scores
 *   - Dynamic load balancing across available experts
 *   - Efficient routing table management and caching
 *
 * Activation Functions:
 *   - SwiGLU: Standard MoE activation with gating
 *   - GELU: Gaussian Error Linear Unit for transformer variants
 *   - ReLU: Rectified Linear Unit for efficiency-focused models
 *   - Custom: User-defined activation functions
 *
 * Launch Configurations:
 *   - Persistent kernels for high-throughput scenarios
 *   - Dynamic scheduling for variable expert workloads
 *   - Memory-bandwidth optimized configurations
 *
 * Optimization Strategies:
 * ========================
 * Expert Load Balancing:
 *   - Adaptive workload distribution based on routing patterns
 *   - Overflow handling for popular experts
 *   - Efficient padding strategies for sparse experts
 *
 * Memory Management:
 *   - Optimized expert weight caching strategies
 *   - Intermediate result buffering between stages
 *   - Minimal memory footprint for large expert ensembles
 *
 * Usage Examples:
 * ===============
 * ```cpp
 * // Create MoE GEMM code generator
 * MoeGemmCodeGen codegen;
 * codegen.problem_ = moe_problem;
 * codegen.tile_desc_ = optimized_tile;
 * codegen.num_experts_ = 8;
 * codegen.act_ = ActivationEnum::SwiGLU;
 * 
 * // Generate optimized kernel code
 * std::string kernel_code = codegen.Emit();
 * std::string instance_name = codegen.GetInstanceName();
 * ```
 */
class MoeGemmCodeGen {
public:
    /**
     * @brief Generate unique instance identifier combining all MoE configuration parameters
     * @return Comprehensive string identifier for profiling, caching, and debugging
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate complete optimized GPU kernel code for this MoE GEMM configuration
     * @return String containing full kernel implementation with expert routing and dual-stage optimization
     */
    std::string Emit() const;

    // ====================== Core Operation Configuration ======================
    
    TopKSoftmaxProblem problem_;  ///< Complete MoE problem specification (routing, dimensions, types)

    // Dual-stage hierarchical tiling strategy
    MoeGemmTileDesc tile_desc_;   ///< Complete tile configuration for optimal MoE performance

    // ====================== Expert Configuration ======================
    
    int64_t num_experts_;         ///< Total number of experts in the MoE ensemble

    // ====================== Memory Access Optimization ======================
    
    int issues_pre_col_;          ///< Memory access issues per column for bandwidth optimization
    int bytes_per_issue_;         ///< Bytes per memory issue for alignment control

    // ====================== Launch Configuration ======================
    
    int launch_type_;             ///< Launch strategy (persistent, dynamic, etc.)
    int64_t block_size_;          ///< Thread block size for GPU kernel execution
    int min_block_per_cu_;        ///< Minimum blocks per compute unit for occupancy control

    // ====================== Activation Function ======================
    
    ActivationEnum act_;          ///< Activation function for MoE computation (SwiGLU, GELU, etc.)
};

}  // namespace flashck

};

}  // namespace flashck