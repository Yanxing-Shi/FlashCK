#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_problem.h"
#include "core/profiling/moe/moe_smooth_quant/moe_smooth_quant_codegen.h"
#include "core/utils/json_config.h"

namespace flashck {

/**
 * @class MoeSmoothQuantEmitter
 * @brief Manages MoE smooth quantization code generation and optimization
 *
 * This class provides comprehensive functionality for MoE smooth quantization operations:
 * - Supports three configuration types: backup (pre-validated), default (parameter ranges), user (custom)
 * - Implements three execution modes: heuristic (0), autotuning (1), hybrid (2)
 * - Generates optimized kernel instances with thread-level tiling
 * - Provides intelligent filtering for smooth quantization optimization
 * 
 * Smooth Quantization Overview:
 * - Applies scaling factors to reduce activation quantization errors
 * - Optimizes memory access patterns for quantization scales
 * - Supports both single-pass and two-pass quantization strategies
 * - Enables efficient INT8 inference with minimal accuracy loss
 * 
 * Usage Patterns:
 * ```cpp
 * auto* emitter = MoeSmoothQuantEmitter::GetInstance();
 * emitter->GenerateInstances(problem);  // Generates all valid instances
 * auto instances = emitter->HeuristicFilter(all_instances, problem);  // Optional filtering
 * ```
 */
class MoeSmoothQuantEmitter {
public:
    MoeSmoothQuantEmitter()  = default;
    ~MoeSmoothQuantEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    MoeSmoothQuantEmitter(const MoeSmoothQuantEmitter&)            = delete;
    MoeSmoothQuantEmitter& operator=(const MoeSmoothQuantEmitter&) = delete;

    /**
     * @brief Get singleton instance of MoeSmoothQuantEmitter
     * @return Pointer to the singleton instance
     */
    static MoeSmoothQuantEmitter* GetInstance()
    {
        static MoeSmoothQuantEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates tile configuration against problem constraints
     * @param tile_desc Tile descriptor containing thread-level parameters
     * @param moe_problem Problem specification including dimensions and data types
     * @return true if tile configuration is valid, false otherwise
     * 
     * Validation includes:
     * - Parameter positivity and alignment constraints
     * - Memory bandwidth optimization for quantization scales
     * - Thread block size limitations
     * - Vector size compatibility with data types
     */
    bool IsValidTile(const MoeSmoothQuantTileDesc& tile_desc, const MoeProblem& moe_problem);

    /**
     * @brief Validates generated code instance
     * @param instance Generated kernel instance to validate
     * @return true if instance is valid and can be compiled
     */
    bool IsValidInstance(const MoeSmoothQuantCodeGen& instance);

    /**
     * @brief Creates kernel instances from configuration
     * @param config Configuration with parameter ranges or single values
     * @param moe_problem Problem specification
     * @return Vector of generated kernel instances
     */
    std::vector<MoeSmoothQuantCodeGen> CreateInstanceForConfig(const MoeSmoothQuantConfig& config, const MoeProblem& moe_problem);

    /**
     * @brief Apply intelligent filtering to reduce search space
     * @param instances All generated instances to filter
     * @param moe_problem Problem specification for context-aware filtering
     * @return Filtered subset of instances with better performance characteristics
     * 
     * Heuristic Strategy:
     * - Prioritizes configurations with optimal memory coalescing
     * - Considers quantization scale access patterns
     * - Balances thread utilization and memory bandwidth
     * - Prefers vector sizes that align with data types
     */
    std::vector<MoeSmoothQuantCodeGen> HeuristicFilter(const std::vector<MoeSmoothQuantCodeGen>& instances, 
                                                      const MoeProblem& moe_problem);

    /**
     * @brief Main instance generation entry point supporting multiple configuration sources
     * @param moe_problem The MoE smooth quantization problem configuration to solve
     * 
     * Execution Strategy (controlled by FC_TUNING_MODE):
     * - Mode 0 (Heuristic): Apply filtering → select optimal subset → random sampling for fast execution
     * - Mode 1 (Autotuning): Generate all valid instances → comprehensive performance search
     * - Mode 2 (Hybrid): Combine heuristic insights with broader search → balanced approach
     * 
     * Configuration Loading (controlled by FC_ENABLE_*_JSON flags):
     * - Backup configs: Pre-validated single-value configurations for immediate deployment
     * - Default configs: Parameter ranges for exploration and tuning
     * - User configs: Custom parameter ranges for specific use cases
     */
    void GenerateInstances(MoeProblem& moe_problem);

    /**
     * @brief Gets the total number of generated instances across all configurations
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const
    {
        return num_instances_;
    }

    /**
     * @brief Get profiling instance map for the given MoE smooth quantization kind
     * @param moe_problem The MoE problem configuration
     * @return Reference to the instance map for the specific operation kind
     */
    std::map<std::string, MoeSmoothQuantCodeGen>& GetInstanceMap(MoeProblem moe_problem)
    {
        GenerateInstances(moe_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    // Instance storage organized by MoE operation type
    std::map<std::string, MoeSmoothQuantCodeGen> instance_map_;

    // Performance tracking
    int64_t num_instances_ = 0;
};

}  // namespace flashck
