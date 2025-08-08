#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <set>
#include <array>

#include "core/profiling/moe/moe_gemm/moe_gemm_codegen.h"

namespace flashck {

// Allowed dual-stage warp distribution combinations for optimal MoE GEMM performance
static const std::vector<std::tuple<int, int, int>> g_moe_gemm_allowed_warp_combinations = {
    {1, 4, 1}, {2, 2, 1}, {4, 1, 1},  // Stage 0: Token-to-Intermediate
    {1, 2, 1}, {2, 1, 1}, {1, 1, 1}   // Stage 1: Intermediate-to-Output (smaller due to expert routing)
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
     * @param moe_gemm_problem MoE problem specification including matrix dimensions, expert count, and data types
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
    bool IsValidTile(const MoeGemmTileDesc& tile_desc, const MoeGemmProblem& moe_gemm_problem);

    /**
     * @brief Apply intelligent filtering to reduce MoE search space with expert routing optimization
     * @param instances All generated MoE instances to filter
     * @param moe_gemm_problem MoE problem specification for context-aware filtering
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
                                               const MoeGemmProblem& moe_gemm_problem);

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
     * @param moe_gemm_problem MoE problem specification
     * @return Vector of generated MoE kernel instances
     */
    std::vector<MoeGemmCodeGen> CreateInstanceForConfig(const flashck::MoeGemmConfig& config, 
                                                       const MoeGemmProblem& moe_gemm_problem);

    /**
     * @brief Generates MoE GEMM operation instances based on the problem specification
     * @param moe_gemm_problem The MoE GEMM problem configuration
     * @return Map of generated MoE operations organized by kind and config name
     */
    void GenerateInstances(MoeGemmProblem& moe_gemm_problem);

    /**
     * @brief Gets the total number of generated MoE instances
     * @return Number of generated MoE instances
     */
    int64_t GetNumInstances() const;

    /**
     * @brief Get profiling instance map for the given MoE kind
     * @param moe_gemm_problem MoE problem specification
     * @return Reference to instance map for the specified MoE kind
     */
    std::map<std::string, MoeGemmCodeGen>& GetInstanceMap(MoeGemmProblem moe_gemm_problem)
    {
        GenerateInstances(moe_gemm_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated MoE instances and resets counters
     */
    void ClearInstances();

private:
    /**
     * @brief Validates expert routing efficiency for given configuration
     * @param tile_desc Tile configuration to validate
     * @param moe_gemm_problem MoE problem with expert routing requirements
     * @return true if routing pattern is efficient
     */
    bool IsValidExpertRouting(const MoeGemmTileDesc& tile_desc, const MoeGemmProblem& moe_gemm_problem);

    /**
     * @brief Validates inter-stage memory bandwidth requirements
     * @param tile_desc Dual-stage tile configuration
     * @param moe_gemm_problem MoE problem specification
     * @return true if bandwidth requirements are satisfied
     */
    bool IsValidInterStageBandwidth(const MoeGemmTileDesc& tile_desc, const MoeGemmProblem& moe_gemm_problem);

    std::map<std::string, MoeGemmCodeGen> instance_map_;
    int64_t num_instances_ = 0;
};

}  // namespace flashck
