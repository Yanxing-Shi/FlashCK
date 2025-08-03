#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <set>
#include <array>

#include "core/profiling/moe/moe_gemm/moe_gemm_codegen.h"
#include "core/profiling/moe/moe_gemm/moe_gemm_problem.h"
#include "core/utils/json_config.h"

namespace flashck {

// Allowed dual-stage warp distribution combinations for optimal MoE GEMM performance
static const std::vector<std::tuple<int, int, int>> g_moe_gemm_allowed_warp_combinations = {
    {1, 4, 1}, {2, 2, 1}, {4, 1, 1},  // Stage 0: Token-to-Intermediate
    {1, 2, 1}, {2, 1, 1}, {1, 1, 1}   // Stage 1: Intermediate-to-Output (smaller due to expert routing)
};

// MoE-specific unsupported combinations: (activation, routing_method, expert_parallel_mode)
const std::set<std::tuple<std::string, std::string, std::string>> g_moe_gemm_unsupported_combinations = {
    {"silu", "hash_routing", "data_parallel"},        // Hash routing conflicts with SiLU
    {"gelu", "capacity_routing", "expert_parallel"},  // Capacity routing incompatible with expert parallelism
    {"relu", "topk_routing", "hybrid_parallel"}       // TopK routing with hybrid parallelism causes memory issues
};

// Architecture and data type specific warp tile combinations for MoE GEMM hardware optimization
static const std::map<std::string, std::map<std::string, std::vector<std::array<int64_t, 6>>>> g_moe_gemm_warp_tile_supported_combinations = {
    {"gfx90a", {
        // Format: {m0_warp_tile, n0_warp_tile, k0_warp_tile, m1_warp_tile, n1_warp_tile, k1_warp_tile}
        {"fp16_fp16_fp16", {{32,32,8,16,16,8},{16,16,16,8,8,16},{32,32,16,16,16,16},{4,64,16,2,32,16}}},
        {"bf16_bf16_bf16", {{32,32,8,16,16,8},{16,16,16,8,8,16},{32,32,16,16,16,16},{4,64,16,2,32,16}}},
        {"fp8_fp8_fp16",   {{32,32,16,16,16,16},{32,32,32,16,16,32}}},
        {"bf8_bf8_fp16",   {{32,32,16,16,16,16},{32,32,32,16,16,32}}}
    }},
    {"gfx942", {
        {"fp16_fp16_fp16", {{32,32,8,16,16,8},{16,16,16,8,8,16},{32,32,16,16,16,16},{4,64,16,2,32,16}}},
        {"bf16_bf16_bf16", {{32,32,8,16,16,8},{16,16,16,8,8,16},{32,32,16,16,16,16},{4,64,16,2,32,16}}},
        {"fp8_fp8_fp16",   {{32,32,16,16,16,16},{32,32,32,16,16,32},{16,16,32,8,8,32},{16,16,64,8,8,64}}},
        {"bf8_bf8_fp16",   {{32,32,16,16,16,16},{32,32,32,16,16,32},{16,16,64,8,8,64},{16,16,32,8,8,32}}},
        {"int8_int8_int32", {{16,16,32,8,8,32},{32,32,16,16,16,16}}}
    }},
    {"gfx950", {
        {"fp16_fp16_fp16", {{32,32,8,16,16,8},{16,16,16,8,8,16},{32,32,16,16,16,16},{4,64,16,2,32,16}}},
        {"bf16_bf16_bf16", {{32,32,8,16,16,8},{16,16,16,8,8,16},{32,32,16,16,16,16},{4,64,16,2,32,16}}},
        {"fp8_fp8_fp16",   {{32,32,16,16,16,16},{32,32,32,16,16,32},{16,16,32,8,8,32},{16,16,64,8,8,64},{16,16,128,8,8,128},{32,32,64,16,16,64}}},
        {"bf8_bf8_fp16",   {{32,32,16,16,16,16},{32,32,32,16,16,32},{16,16,64,8,8,64},{16,16,32,8,8,32},{16,16,128,8,8,128},{32,32,64,16,16,64}}}
    }}
};

/**
 * @class MoeGemmEmitter
 * @brief Manages MoE GEMM operation code generation and optimization
 *
 * This class provides comprehensive functionality for Mixture of Experts (MoE) GEMM operations:
 * - Supports three configuration types: backup (pre-validated), default (parameter ranges), user (custom)
 * - Implements three execution modes: heuristic (0), autotuning (1), hybrid (2)
 * - Generates optimized dual-stage MoE GEMM kernel instances with hierarchical tiling
 * - Provides intelligent filtering for expert routing and activation optimization
 * 
 * MoE Architecture Features:
 * - Dual-stage processing: Stage 0 (Token→Intermediate), Stage 1 (Intermediate→Output)
 * - Expert routing optimization with load balancing and sparse selection
 * - Multi-level tiling: Block-level → Warp-level → Thread-level optimization for both stages
 * - Activation function fusion (SwiGLU, GELU, ReLU) for memory efficiency
 * - Hardware-specific optimizations for different GPU architectures
 * - Expert parallelism strategies: Data parallel, Expert parallel, Hybrid parallel
 * 
 * Performance Optimizations:
 * - Intelligent expert load balancing to minimize workload imbalance
 * - Sparse expert selection with capacity factor optimization
 * - Memory access pattern optimization for expert weights
 * - Pipeline fusion between routing, computation, and activation stages
 * - Comprehensive validation and constraint checking for dual-stage tiling
 * 
 * Usage Example:
 * ```cpp
 * auto* emitter = MoeGemmEmitter::GetInstance();
 * emitter->GenerateInstances(moe_problem);  // Generate all valid MoE instances
 * auto instances = emitter->HeuristicFilter(all_instances, moe_problem);  // Expert routing optimization
 * ```
 */
class MoeGemmEmitter {
public:
    MoeGemmEmitter()  = default;
    ~MoeGemmEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    MoeGemmEmitter(const MoeGemmEmitter&)            = delete;
    MoeGemmEmitter& operator=(const MoeGemmEmitter&) = delete;

    /**
     * @brief Get singleton instance of MoeGemmEmitter
     * @return Pointer to the singleton instance
     */
    static MoeGemmEmitter* GetInstance()
    {
        static MoeGemmEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates dual-stage MoE GEMM tile configuration against problem constraints and hardware limitations
     * @param tile_desc Dual-stage tile descriptor containing block/warp/thread-level parameters for both stages
     * @param moe_problem MoE problem specification including matrix dimensions, expert count, and data types
     * @return true if tile configuration is valid for dual-stage MoE processing, false otherwise
     * 
     * Validation includes:
     * - Parameter positivity and divisibility constraints for both stages
     * - Hardware resource limitations (registers, shared memory) per stage
     * - Architecture-specific warp tile size restrictions for MoE operations
     * - Expert routing memory access pattern optimality
     * - Load balancing constraints across experts
     * - Inter-stage memory bandwidth requirements
     */
    bool IsValidTile(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const;

    /**
     * @brief Apply intelligent filtering to reduce MoE search space with expert routing optimization
     * @param instances All generated MoE instances to filter
     * @param moe_problem MoE problem specification for context-aware filtering
     * @return Filtered subset of instances with better performance characteristics and expert load balancing
     * 
     * MoE Heuristic Strategy:
     * - Prioritizes configurations with optimal expert routing patterns
     * - Considers token distribution and expert capacity factors
     * - Balances register usage and memory bandwidth across both stages
     * - Uses architecture-specific performance models for MoE workloads
     * - Optimizes activation function fusion opportunities
     * - Evaluates expert parallelism efficiency
     */
    std::vector<MoeGemmCodeGen> HeuristicFilter(const std::vector<MoeGemmCodeGen>& instances, 
                                               const MoeProblem& moe_problem);

    /**
     * @brief Validates MoE-specific combination of activation, routing, and parallelism
     * @param activation Activation function type (SwiGLU, GELU, ReLU, etc.)
     * @param routing_method Expert routing method (TopK, Capacity, Hash, etc.)
     * @param expert_parallel_mode Expert parallelism strategy (Data, Expert, Hybrid)
     * @return true if combination is supported by MoE architecture and hardware
     */
    bool IsValidMoeCombination(const std::string& activation, const std::string& routing_method, 
                              const std::string& expert_parallel_mode);

    /**
     * @brief Validates generated MoE code instance
     * @param instance Generated MoE kernel instance to validate
     * @return true if instance is valid and can be compiled for MoE execution
     */
    bool IsValidInstance(const MoeGemmCodeGen& instance);

    /**
     * @brief Creates MoE kernel instances from configuration
     * @param config Configuration with parameter ranges or single values for MoE
     * @param moe_problem MoE problem specification
     * @return Vector of generated MoE kernel instances
     */
    std::vector<MoeGemmCodeGen> CreateInstanceForConfig(const flashck::MoeGemmConfig& config, 
                                                       const MoeProblem& moe_problem);

    /**
     * @brief Generates MoE GEMM operation instances based on the problem specification
     * @param moe_problem The MoE GEMM problem configuration
     * @return Map of generated MoE operations organized by kind and config name
     */
    void GenerateInstances(MoeProblem& moe_problem);

    /**
     * @brief Gets the total number of generated MoE instances
     * @return Number of generated MoE instances
     */
    int64_t GetNumInstances() const;

    /**
     * @brief Get profiling instance map for the given MoE kind
     * @param moe_problem MoE problem specification
     * @return Reference to instance map for the specified MoE kind
     */
    std::map<std::string, MoeGemmCodeGen>& GetInstanceMap(MoeProblem moe_problem)
    {
        GenerateInstances(moe_problem);
        return instance_map_[moe_problem.kind_];
    }

    /**
     * @brief Clears all generated MoE instances and resets counters
     */
    void ClearInstances();

private:
    /**
     * @brief Validates expert routing efficiency for given configuration
     * @param tile_desc Tile configuration to validate
     * @param moe_problem MoE problem with expert routing requirements
     * @return true if routing pattern is efficient
     */
    bool IsValidExpertRouting(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const;

    /**
     * @brief Validates load balancing constraints across experts
     * @param tile_desc Tile configuration to validate
     * @param moe_problem MoE problem with load balancing requirements
     * @return true if load balancing is optimal
     */
    bool IsValidLoadBalancing(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const;

    /**
     * @brief Validates inter-stage memory bandwidth requirements
     * @param tile_desc Dual-stage tile configuration
     * @param moe_problem MoE problem specification
     * @return true if bandwidth requirements are satisfied
     */
    bool IsValidInterStageBandwidth(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const;

    std::map<MoeGemmKind, std::map<std::string, MoeGemmCodeGen>> instance_map_;
    int64_t                                                      num_instances_ = 0;
};

}  // namespace flashck
