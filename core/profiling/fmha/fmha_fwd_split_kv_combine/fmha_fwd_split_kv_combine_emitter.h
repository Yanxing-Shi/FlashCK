#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/fmha/fmha_library.h"
#include "core/profiling/fmha/fmha_problem.h"

#include "core/profiling/fmha/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_codegen.h"
#include "core/profiling/json_config.h"

FC_DECLARE_int32(FC_TUNING_MODE);         // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);  // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON); // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);    // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);  // Base path for config files

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
 * Configuration Management:
 * ========================
 * Uses modern FLAGS-based configuration system:
 *   FC_TUNING_MODE: 0=heuristic, 1=autotuning, 2=hybrid
 *   FC_ENABLE_BACKUP_JSON: Load backup_config.json (pre-validated configurations)
 *   FC_ENABLE_DEFAULT_JSON: Load default_config.json (parameter ranges)
 *   FC_ENABLE_USER_JSON: Load user_config.json (custom configurations)
 *   FC_CONFIG_JSON_PATH: Base directory for configuration files
 *
 * Configuration Structure:
 * ========================
 * - backup_config.json: Array of pre-validated single configurations
 * - default_config.json: Single configuration with parameter ranges
 * - user_config.json: Single configuration with custom parameter ranges
 *
 * Tuning Strategy:
 * ========================
 * Heuristic Mode (0): Fast execution with intelligent filtering
 *   - Apply split KV combine-specific heuristics
 *   - Random selection from filtered candidates
 *   - Optimized for production deployment
 *
 * Autotuning Mode (1): Comprehensive search space exploration
 *   - Use all valid instances for profiling
 *   - Maximum performance potential
 *   - Best for offline optimization
 *
 * Hybrid Mode (2): Balanced approach
 *   - Combine heuristic filtering with expanded search
 *   - Remove duplicates intelligently
 *   - Good compromise between speed and thoroughness
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
     * 
     * Performs comprehensive validation for split KV combine operations:
     * - Ensures all tile dimensions are positive and well-formed
     * - Validates warp and thread-level tiling constraints
     * - Checks block size divisibility requirements
     * - Verifies compatibility with problem dimensions
     * - Applies split KV combine-specific constraints
     * 
     * @param tile_desc Tile descriptor to validate
     * @param fmha_problem Problem specification for validation context
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const FmhaFwdSplitKVCombineTileDesc& tile_desc, const FmhaProblem& fmha_problem);

    /**
     * @brief Validate complete instance configuration
     * 
     * Validates the entire instance including tile descriptor and additional parameters:
     * - Calls IsValidTile for core tile validation
     * - Checks launch configuration parameters
     * - Validates pipeline and memory access patterns
     * 
     * @param instance Instance to validate
     * @return true if instance is valid, false otherwise
     */
    bool IsValidInstance(const FmhaFwdSplitKVCombineCodeGen& instance);

    /**
     * @brief Apply intelligent heuristic filtering to reduce search space
     * 
     * Split KV combine-specific heuristics:
     * - Optimize for efficient result aggregation patterns
     * - Filter configurations with poor split result combination efficiency
     * - Prioritize memory bandwidth utilization for reduction operations
     * - Balance register usage for combine operations
     * - Ensure optimal block and warp utilization for reduction
     * 
     * @param instances Vector of instances to filter
     * @param fmha_problem Problem context for filtering decisions
     * @return Filtered vector of high-quality instances optimized for split KV combine
     */
    std::vector<FmhaFwdSplitKVCombineCodeGen> HeuristicFilter(const std::vector<FmhaFwdSplitKVCombineCodeGen>& instances, 
                                                              const FmhaProblem& fmha_problem);

    /**
     * @brief Create instances from configuration specification using CartesianProduct
     * 
     * Generates all combinations of parameters from the configuration ranges:
     * - Block tile configuration (6 dimensions: m0, n0, k0, k0_max, n1, k1)
     * - Block warp configuration (6 dimensions: m0, n0, k0, m1, n1, k1)
     * - Warp tile configuration (6 dimensions: m0, n0, k0, m1, n1, k1)
     * - Padding configuration (4 boolean flags: s, sk, d, dv)
     * - Launch configuration (min_block_per_cu)
     * - Pipeline configuration (pipeline enum)
     * 
     * Uses CartesianProduct utility to generate all valid parameter combinations.
     * 
     * @param config Configuration with parameter ranges or fixed values
     * @param fmha_problem Problem specification for context
     * @return Vector of generated instances
     */
    std::vector<FmhaFwdSplitKVCombineCodeGen> CreateInstanceForConfig(const FmhaFwdSplitKVCombineConfig& config, const FmhaProblem& fmha_problem);

    /**
     * @brief Generate optimized FMHA split KV combine instances using multi-source configuration and intelligent filtering
     * 
     * Modern implementation using FLAGS-based configuration system and LoadConfigJson utility:
     * 
     * Configuration Loading Strategy:
     * 1. Load backup_config.json if FC_ENABLE_BACKUP_JSON (pre-validated single configurations)
     * 2. Load default_config.json if FC_ENABLE_DEFAULT_JSON (parameter ranges for search)
     * 3. Load user_config.json if FC_ENABLE_USER_JSON (custom parameter ranges)
     * 
     * Tuning Modes (controlled by FC_TUNING_MODE):
     * 0 = Heuristic: Apply split KV combine-specific filtering + random selection (fastest)
     *     - Filter for optimal combine operation patterns
     *     - Random selection from filtered candidates
     *     - Optimized for production deployment
     * 
     * 1 = Autotuning: Use all valid instances for comprehensive search
     *     - Maximum search space coverage
     *     - All valid configurations tested
     *     - Best for offline optimization
     * 
     * 2 = Hybrid: Apply heuristic filtering but keep broader candidate set
     *     - Combine heuristic filtering with expanded search
     *     - Remove duplicates intelligently
     *     - Good compromise between speed and thoroughness
     * 
     * Error Handling:
     * - Validates tuning mode range (0-2)
     * - Handles JSON loading failures gracefully
     * - Ensures at least one valid instance exists
     * - Provides detailed logging for debugging
     * 
     * @param fmha_problem The FMHA problem configuration and constraints
     */
    void GenerateInstances(FmhaProblem& fmha_problem);

    /**
     * @brief Gets the total number of generated instances across all FMHA kinds
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const
    {
        return num_instances_;
    }

    /**
     * @brief Get profiling instance map for the given FMHA kind
     * 
     * Automatically triggers instance generation if not already done for this kind.
     * Returns reference to allow direct manipulation of the instance map.
     * 
     * @param fmha_problem The FMHA problem configuration
     * @return Reference to the instance map for the specific FMHA kind
     */
    std::map<std::string, FmhaFwdSplitKVCombineCodeGen>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated instances and resets counters
     * 
     * Useful for testing, re-initialization, or memory cleanup.
     * Resets both the instance map and the total instance counter.
     */
    void ClearInstances();

private:
    // Maps FmhaKind to instance name -> instance mapping
    // Allows different FMHA kinds to have their own instance sets
    std::map<std::string, FmhaFwdSplitKVCombineCodeGen> instance_map_;

    // Total number of instances generated across all kinds
    int64_t num_instances_ = 0;
};

}  // namespace flashck
