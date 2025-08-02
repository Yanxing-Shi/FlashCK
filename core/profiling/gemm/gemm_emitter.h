#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/profiling/gemm/gemm_codegen.h"
#include "core/profiling/gemm/gemm_problem.h"
#include "core/utils/json_config.h"

namespace flashck {

// Allowed warp distribution combinations for optimal performance
static const std::vector<std::tuple<int, int, int>> g_tile_gemm_allowed_warp_combinations = {
        {1, 4, 1}, {2, 2, 1}, {4, 1, 1}
};

// Set of unsupported combinations: (pipeline, epilogue, scheduler)
const std::set<std::tuple<PipelineVersionEnum, EpilogueEnum, PipelineSchedulerEnum>> g_tile_gemm_unsupported_combinations = {
    {GetPipelineVersionEnumFromString("compv3"), GetEpilogueEnumFromString("cshuffle"), GetPipelineSchedulerEnumFromString("interwave")},
    {GetPipelineVersionEnumFromString("compv3"), GetEpilogueEnumFromString("default"),  GetPipelineSchedulerEnumFromString("interwave")},
    {GetPipelineVersionEnumFromString("compv4"), GetEpilogueEnumFromString("cshuffle"), GetPipelineSchedulerEnumFromString("interwave")},
    {GetPipelineVersionEnumFromString("compv4"), GetEpilogueEnumFromString("default"),  GetPipelineSchedulerEnumFromString("interwave")}
};

// Architecture and data type specific warp tile combinations for hardware optimization
static const std::map<std::string, std::map<std::string, std::vector<std::array<int64_t, 3>>>> g_tile_gemm_warp_tile_supported_combinations = {
    {"gfx90a", {
        {"fp16_fp16_fp16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"bf16_bf16_bf16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"fp8_fp8_fp16",   {{32,32,16},{32,32,32}}},
        {"bf8_bf8_fp16",   {{32,32,16},{32,32,32}}}
    }},
    {"gfx942", {
        {"fp16_fp16_fp16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"bf16_bf16_bf16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"fp8_fp8_fp16",   {{32,32,16},{32,32,32},{16,16,32},{16,16,64}}},
        {"bf8_bf8_fp16",   {{32,32,16},{32,32,32},{16,16,64},{16,16,32}}},
        {"int8_int8_int32", {{16,16,32},{32,32,16}}}
    }},
    {"gfx950", {
        {"fp16_fp16_fp16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"bf16_bf16_bf16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"fp8_fp8_fp16",   {{32,32,16},{32,32,32},{16,16,32},{16,16,64},{16,16,128},{32,32,64}}},
        {"bf8_bf8_fp16",   {{32,32,16},{32,32,32},{16,16,64},{16,16,32},{16,16,128},{32,32,64}}}
    }}
};



/**
 * @class GemmEmitter
 * @brief Manages GEMM operation code generation and optimization
 *
 * This class provides comprehensive functionality for GEMM operations:
 * - Supports three configuration types: backup (pre-validated), default (parameter ranges), user (custom)
 * - Implements three execution modes: heuristic (0), autotuning (1), hybrid (2)
 * - Generates optimized kernel instances with hierarchical tiling
 * - Provides intelligent filtering for performance optimization
 * 
 * Architecture Features:
 * - Multi-level tiling: Block-level → Warp-level → Thread-level optimization
 * - Hardware-specific optimizations for different GPU architectures
 * - Pipeline and epilogue fusion for memory efficiency
 * - Comprehensive validation and constraint checking
 * 
 * Usage Example:
 * ```cpp
 * auto* emitter = GemmEmitter::GetInstance();
 * emitter->GenerateInstances(problem);  // Generate all valid instances
 * auto instances = emitter->HeuristicFilter(all_instances, problem);  // Optional filtering
 * ```
 */
class GemmEmitter {
public:
    GemmEmitter()  = default;
    ~GemmEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    GemmEmitter(const GemmEmitter&)            = delete;
    GemmEmitter& operator=(const GemmEmitter&) = delete;

    /**
     * @brief Get singleton instance of GemmEmitter
     * @return Pointer to the singleton instance
     */
    static GemmEmitter* GetInstance()
    {
        static GemmEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates tile configuration against problem constraints and hardware limitations
     * @param tile_desc Tile descriptor containing block/warp/thread-level parameters
     * @param gemm_problem Problem specification including matrix dimensions and data types
     * @return true if tile configuration is valid, false otherwise
     * 
     * Validation includes:
     * - Parameter positivity and divisibility constraints
     * - Hardware resource limitations (registers, shared memory)
     * - Architecture-specific warp tile size restrictions
     * - Memory access pattern optimality
     */
    bool IsValidTile(const GemmTileDesc& tile_desc, const GemmProblem& gemm_problem) const;

    /**
     * @brief Apply intelligent filtering to reduce search space
     * @param instances All generated instances to filter
     * @param gemm_problem Problem specification for context-aware filtering
     * @return Filtered subset of instances with better performance characteristics
     * 
     * Heuristic Strategy:
     * - Prioritizes configurations with optimal memory access patterns
     * - Considers matrix dimensions and data types
     * - Balances register usage and memory bandwidth
     * - Uses architecture-specific performance models
     */
    std::vector<GemmCodeGen> HeuristicFilter(const std::vector<GemmCodeGen>& instances, 
                                            const GemmProblem& gemm_problem);

    /**
     * @brief Validates pipeline, epilogue, and scheduler combination
     * @param pipeline Pipeline version (mem, compv3, compv4, etc.)
     * @param epilogue Epilogue type (default, cshuffle, etc.)
     * @param scheduler Pipeline scheduler (interwave, intrawave, etc.)
     * @return true if combination is supported by hardware and library
     */
    bool IsValidCombination(const PipelineVersionEnum& pipeline, const EpilogueEnum& epilogue, const PipelineSchedulerEnum& scheduler);

    /**
     * @brief Validates generated code instance
     * @param instance Generated kernel instance to validate
     * @return true if instance is valid and can be compiled
     */
    bool IsValidInstance(const GemmCodeGen& instance);

    /**
     * @brief Creates kernel instances from configuration
     * @param config Configuration with parameter ranges or single values
     * @param gemm_problem Problem specification
     * @return Vector of generated kernel instances
     */
    std::vector<GemmCodeGen> CreateInstanceForConfig(const flashck::GemmConfig& config, const GemmProblem& gemm_problem);

    /**
     * @brief Generates gemm operation instances based on the problem specification
     * @param gemm_problem The gemm problem configuration
     * @return Map of generated gemm operations organized by kind and config name
     */
    void GenerateInstances(GemmProblem& gemm_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const;

    // get profiling instance map for the given norm kind
    std::map<std::string, GemmCodeGen>& GetInstanceMap(GemmProblem gemm_problem)
    {
        GenerateInstances(gemm_problem);
        return instance_map_[gemm_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:

    std::map<GemmKind, std::map<std::string, GemmCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};


}  // namespace flashck