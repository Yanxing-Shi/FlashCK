
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/fmha/fmha_library.h"
#include "core/profiling/fmha/fmha_problem.h"
#include "core/profiling/fmha/fmha_fwd/fmha_fwd_codegen.h"
#include "core/utils/json_config.h"

namespace flashck {

/**
 * @class FmhaFwdEmitter
 * @brief Manages FMHA forward pass code generation and optimization
 *
 * This class provides comprehensive functionality for FMHA forward operations:
 * - Supports three configuration types: backup (pre-validated), default (parameter ranges), user (custom)
 * - Implements three execution modes: heuristic (0), autotuning (1), hybrid (2)
 * - Generates optimized kernel instances with hierarchical tiling
 * - Provides intelligent filtering for performance optimization
 * 
 * Architecture Overview:
 * - Block-level tiling: Divides computation across CUDA thread blocks
 * - Warp-level distribution: Assigns work within each thread block
 * - Thread-level optimization: Fine-grained vectorization and memory access
 * 
 * Usage Patterns:
 * ```cpp
 * auto* emitter = FmhaFwdEmitter::GetInstance();
 * emitter->GenerateInstances(problem);  // Generates all valid instances
 * auto instances = emitter->HeuristicFilter(all_instances, problem);  // Optional filtering
 * ```
 */
class FmhaFwdEmitter {
public:
    FmhaFwdEmitter()  = default;
    ~FmhaFwdEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    FmhaFwdEmitter(const FmhaFwdEmitter&)            = delete;
    FmhaFwdEmitter& operator=(const FmhaFwdEmitter&) = delete;

    /**
     * @brief Get singleton instance of FmhaFwdEmitter
     * @return Pointer to the singleton instance
     */
    static FmhaFwdEmitter* GetInstance()
    {
        static FmhaFwdEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates tile configuration against problem constraints
     * @param tile_desc Tile descriptor containing block/warp/thread-level parameters
     * @param fmha_problem Problem specification including dimensions and data types
     * @return true if tile configuration is valid, false otherwise
     * 
     * Validation includes:
     * - Parameter positivity and divisibility constraints
     * - Hardware resource limitations (registers, shared memory)
     * - Mathematical correctness for attention computation
     */
    bool IsValidTile(const FmhaFwdTileDesc& tile_desc, const FmhaProblem& fmha_problem);

    /**
     * @brief Validates generated code instance
     * @param instance Generated kernel instance to validate
     * @return true if instance is valid and can be compiled
     */
    bool IsValidInstance(const FmhaFwdCodeGen& instance);

    /**
     * @brief Creates kernel instances from configuration
     * @param config Configuration with parameter ranges or single values
     * @param fmha_problem Problem specification
     * @return Vector of generated kernel instances
     */
    std::vector<FmhaFwdCodeGen> CreateInstanceForConfig(const FmhaFwdConfig& config, const FmhaProblem& fmha_problem);

    /**
     * @brief Apply intelligent filtering to reduce search space
     * @param instances All generated instances to filter
     * @param fmha_problem Problem specification for context-aware filtering
     * @return Filtered subset of instances with better performance characteristics
     * 
     * Heuristic Strategy:
     * - Prioritizes configurations with optimal memory access patterns
     * - Considers problem size and hardware characteristics
     * - Balances register usage and memory bandwidth
     * - Uses performance models to predict efficiency
     */
    std::vector<FmhaFwdCodeGen> HeuristicFilter(const std::vector<FmhaFwdCodeGen>& instances, 
                                               const FmhaProblem& fmha_problem);

    /**
     * @brief Main instance generation entry point supporting multiple configuration sources
     * @param fmha_problem The FMHA problem configuration to solve
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
    void GenerateInstances(FmhaProblem& fmha_problem);

    /**
     * @brief Gets the total number of generated instances across all configurations
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const
    {
        return num_instances_;
    }

    /**
     * @brief Get profiling instance map for the given FMHA kind
     * @param fmha_problem The FMHA problem configuration
     * @return Reference to the instance map for the specific FMHA kind
     */
    std::map<std::string, FmhaFwdCodeGen>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_[fmha_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    // Instance storage organized by FMHA operation type
    std::map<FmhaKind, std::map<std::string, FmhaFwdCodeGen>> instance_map_;
    
    // Performance tracking
    int64_t num_instances_ = 0;
};

}  // namespace flashck