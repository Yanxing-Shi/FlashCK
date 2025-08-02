
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/fmha/fmha_library.h"
#include "core/profiling/fmha/fmha_problem.h"

#include "core/profiling/fmha/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_codegen.h"
#include "core/profiling/fmha/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_backup_config.h"

namespace flashck {

/**
 * @class FmhaFwdSplitKVCombineEmitter
 * @brief High-performance FMHA forward split KV combine operation instance generator with intelligent optimization
 *
 * This class provides comprehensive FMHA split KV combine instance generation and management with support for:
 * - Three configuration sources: backup_config.json (pre-validated), default_config.json (ranges), user_config.json (custom)
 * - Three tuning modes: heuristic (fast), autotuning (comprehensive), hybrid (balanced)
 * - Hardware-specific tile optimization and validation
 * - Intelligent filtering for optimal performance
 *
 * Architecture Features:
 * ========================
 * Split KV Combine Processing:
 *   - Efficient combination of split key-value results from parallel attention computations
 *   - Optimized reduction and aggregation patterns for multi-stage attention
 *   - Support for variable-length sequence aggregation with minimal overhead
 *
 * Hierarchical Tiling:
 *   Block Level (GPU CU/SM): Controls memory bandwidth and occupancy
 *   Warp Level (GPU warp): Manages SIMD execution and data locality  
 *   Thread Level (GPU thread): Optimizes register usage and instruction throughput
 *
 * Configuration Structure:
 * ========================
 * - backup_config.json: Array of pre-validated single configurations
 * - default_config.json: Single configuration with parameter ranges
 * - user_config.json: Single configuration with custom parameter ranges
 */
class FmhaFwdSplitKVCombineEmitter {
public:
    FmhaFwdSplitKVCombineEmitter()  = default;
    ~FmhaFwdSplitKVCombineEmitter() = default;

    // Enforce singleton pattern with deleted copy operations
    FmhaFwdSplitKVCombineEmitter(const FmhaFwdSplitKVCombineEmitter&)            = delete;
    FmhaFwdSplitKVCombineEmitter& operator=(const FmhaFwdSplitKVCombineEmitter&) = delete;

    /**
     * @brief Thread-safe singleton instance access
     * @return Pointer to the singleton instance
     */
    static FmhaFwdSplitKVCombineEmitter* GetInstance()
    {
        static FmhaFwdSplitKVCombineEmitter instance;
        return &instance;
    }

    /**
     * @brief Validate tile descriptor against problem constraints and hardware limitations
     * @param tile_desc Tile descriptor to validate
     * @param fmha_problem Problem specification for validation context
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const FmhaFwdSplitKVCombineTileDesc& tile_desc, const FmhaProblem& fmha_problem);

    /**
     * @brief Validate complete instance configuration
     * @param instance Instance to validate
     * @return true if instance is valid, false otherwise
     */
    bool IsValidInstance(const FmhaFwdSplitKVCombineCodeGen& instance);

    /**
     * @brief Apply intelligent heuristic filtering to reduce search space
     * @param instances Vector of instances to filter
     * @param fmha_problem Problem context for filtering decisions
     * @return Filtered vector of high-quality instances
     */
    std::vector<FmhaFwdSplitKVCombineCodeGen> HeuristicFilter(const std::vector<FmhaFwdSplitKVCombineCodeGen>& instances, 
                                                              const FmhaProblem& fmha_problem);

    /**
     * @brief Create instances from configuration specification
     * @param config Configuration with parameter ranges or fixed values
     * @param fmha_problem Problem specification for context
     * @return Vector of generated instances
     */
    std::vector<FmhaFwdSplitKVCombineCodeGen> CreateInstanceForConfig(const FmhaFwdSplitKVCombineConfig& config, const FmhaProblem& fmha_problem);

    /**
     * @brief Generate optimized FMHA split KV combine instances using multi-source configuration and intelligent filtering
     * 
     * Configuration Loading Priority:
     * 1. backup_config.json - Pre-validated single configurations (highest priority)
     * 2. default_config.json - Parameter ranges for comprehensive search
     * 3. user_config.json - Custom parameter ranges (user overrides)
     * 
     * Tuning Modes (controlled by FC_TUNING_MODE):
     * 0 = Heuristic: Apply intelligent filtering + random selection (fastest)
     * 1 = Autotuning: Use all valid instances for comprehensive search
     * 2 = Hybrid: Apply heuristic filtering but keep broader candidate set
     * 
     * @param fmha_problem The FMHA problem configuration and constraints
     */
     * @brief Generates FMHA operation instances based on the problem specification
     * @param fmha_problem The FMHA problem configuration
     */
    void GenerateInstances(FmhaProblem& fmha_problem);

    /**
     * @brief Gets the total number of generated instances
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
    std::map<std::string, FmhaFwdSplitKVCombineCodeGen>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_[fmha_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    
    std::map<FmhaKind, std::map<std::string, FmhaFwdSplitKVCombineCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck