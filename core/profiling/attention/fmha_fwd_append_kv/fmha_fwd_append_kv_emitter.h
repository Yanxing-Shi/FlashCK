
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/attention/fmha_library.h"
#include "core/profiling/attention/fmha_fwd_append_kv/fmha_fwd_append_kv_codegen.h"

namespace flashck {

/**
 * @class FmhaFwdAppendKVEmitter
 * @brief High-performance FMHA forward append KV operation instance generator with intelligent optimization
 *
 * This class provides comprehensive FMHA instance generation and management with support for:
 * - Three configuration sources: backup_config.json (pre-validated), default_config.json (ranges), user_config.json (custom)
 * - Three tuning modes: heuristic (fast), autotuning (comprehensive), hybrid (balanced)
 * - Hardware-specific tile optimization and validation
 * - Intelligent filtering for optimal performance
 *
 * Architecture Features:
 * ========================
 * Hierarchical Tiling:
 *   Block Level (GPU CU/SM): Controls memory bandwidth and occupancy
 *   Warp Level (GPU warp): Manages SIMD execution and data locality  
 *   Thread Level (GPU thread): Optimizes register usage and instruction throughput
 *
 * Padding Support:
 *   - Query sequence padding (s): Handle variable Q sequence lengths
 *   - Key-value sequence padding (sk): Handle variable KV sequence lengths
 *   - QK head dimension padding (d): Optimize for non-power-of-2 head dimensions
 *   - V head dimension padding (dv): Handle mismatched V head dimensions
 *
 * Pipeline Configurations:
 *   - QKVS: Query-Key-Value-Softmax pipeline
 *   - QKV: Query-Key-Value pipeline (without softmax)
 *   - Custom: Application-specific pipeline configurations
 *
 * Usage Examples:
 * ===============
 * ```cpp
 * // Get singleton instance
 * auto* emitter = FmhaFwdAppendKVEmitter::GetInstance();
 * 
 * // Generate instances for problem
 * FmhaFwdAppendKVProblem problem = {...};
 * emitter->GenerateInstances(problem);
 * 
 * // Get generated instances
 * auto& instances = emitter->GetInstanceMap(problem);
 * ```
 *
 * Configuration Structure:
 * ========================
 * - backup_config.json: Array of pre-validated single configurations
 * - default_config.json: Single configuration with parameter ranges
 * - user_config.json: Single configuration with custom parameter ranges
 */
class FmhaFwdAppendKVEmitter {
public:
    FmhaFwdAppendKVEmitter()  = default;
    ~FmhaFwdAppendKVEmitter() = default;

    // Enforce singleton pattern with deleted copy operations
    FmhaFwdAppendKVEmitter(const FmhaFwdAppendKVEmitter&)            = delete;
    FmhaFwdAppendKVEmitter& operator=(const FmhaFwdAppendKVEmitter&) = delete;

    /**
     * @brief Thread-safe singleton instance access
     * @return Pointer to the singleton instance
     */
    static FmhaFwdAppendKVEmitter* GetInstance()
    {
        static FmhaFwdAppendKVEmitter instance;
        return &instance;
    }

    /**
     * @brief Validate tile descriptor against problem constraints and hardware limitations
     * @param tile_desc Tile descriptor to validate
     * @param fmha_fwd_append_kv_problem Problem specification for validation context
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const FmhaFwdAppendKVTileDesc& tile_desc, const FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem);

    /**
     * @brief Validate complete instance configuration
     * @param instance Instance to validate
     * @return true if instance is valid, false otherwise
     */
    bool IsValidInstance(const FmhaFwdAppendKVCodeGen& instance);

    /**
     * @brief Apply intelligent heuristic filtering to reduce search space
     * @param instances Vector of instances to filter
     * @param fmha_fwd_append_kv_problem Problem context for filtering decisions
     * @return Filtered vector of high-quality instances
     */
    std::vector<FmhaFwdAppendKVCodeGen> HeuristicFilter(const std::vector<FmhaFwdAppendKVCodeGen>& instances, 
                                                        const FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem);

    /**
     * @brief Create instances from configuration specification
     * @param config Configuration with parameter ranges or fixed values
     * @param fmha_fwd_append_kv_problem Problem specification for context
     * @return Vector of generated instances
     */
    std::vector<FmhaFwdAppendKVCodeGen> CreateInstanceForConfig(const FmhaFwdAppendKVConfig& config, const FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem);

    /**
     * @brief Generate optimized FMHA instances using multi-source configuration and intelligent filtering
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
     * @param fmha_fwd_append_kv_problem The FMHA problem configuration and constraints
     */
    void GenerateInstances(FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem);

    /**
     * @brief Get total number of generated valid instances
     * @return Number of generated instances across all FMHA kinds
     */
    int64_t GetNumInstances() const
    {
        return num_instances_;
    }

    /**
     * @brief Get profiling instance map for the given FMHA kind
     * @param fmha_fwd_append_kv_problem The FMHA problem configuration
     * @return Reference to the instance map for the specific FMHA kind
     */
    std::map<std::string, FmhaFwdAppendKVCodeGen>& GetInstanceMap(FmhaFwdAppendKVProblem fmha_fwd_append_kv_problem)
    {
        GenerateInstances(fmha_fwd_append_kv_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:

    std::map<std::string, FmhaFwdAppendKVCodeGen> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck