
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/attention/fmha_library.h"
#include "core/profiling/attention/fmha_problem.h"
#include "core/profiling/attention/fmha_paged_kv_prefill/fmha_paged_kv_prefill_codegen.h"

namespace flashck {

/**
 * @class FmhaPagedKVPrefillEmitter
 * @brief High-performance FMHA paged KV prefill operation instance generator with intelligent optimization
 *
 * This class provides comprehensive FMHA paged KV prefill instance generation and management with support for:
 * - Three configuration sources: backup_config.json (pre-validated), default_config.json (ranges), user_config.json (custom)
 * - Three tuning modes: heuristic (fast), autotuning (comprehensive), hybrid (balanced)
 * - Hardware-specific tile optimization and validation
 * - Intelligent filtering for optimal performance
 *
 * Architecture Features:
 * ========================
 * Paged KV Cache Management:
 *   - Efficient paged memory access patterns for large KV caches
 *   - Optimized prefetching strategies for variable-length sequences
 *   - Support for dynamic memory allocation and deallocation in inference
 *   - Hardware-aware page size optimization for optimal cache utilization
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
class FmhaPagedKVPrefillEmitter {
public:
    FmhaPagedKVPrefillEmitter()  = default;
    ~FmhaPagedKVPrefillEmitter() = default;

    // Enforce singleton pattern with deleted copy operations
    FmhaPagedKVPrefillEmitter(const FmhaPagedKVPrefillEmitter&)            = delete;
    FmhaPagedKVPrefillEmitter& operator=(const FmhaPagedKVPrefillEmitter&) = delete;

    /**
     * @brief Thread-safe singleton instance access
     * @return Pointer to the singleton instance
     */
    static FmhaPagedKVPrefillEmitter* GetInstance()
    {
        static FmhaPagedKVPrefillEmitter instance;
        return &instance;
    }

    /**
     * @brief Validate tile descriptor against problem constraints and hardware limitations
     * @param tile_desc Tile descriptor to validate
     * @param fmha_problem Problem specification for validation context
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const FmhaPagedKVPrefillTileDesc& tile_desc, const FmhaProblem& fmha_problem);

    /**
     * @brief Validate complete instance configuration
     * @param instance Instance to validate
     * @return true if instance is valid, false otherwise
     */
    bool IsValidInstance(const FmhaPagedKVPrefillCodeGen& instance);

    /**
     * @brief Apply intelligent heuristic filtering to reduce search space
     * @param instances Vector of instances to filter
     * @param fmha_problem Problem context for filtering decisions
     * @return Filtered vector of high-quality instances
     */
    std::vector<FmhaPagedKVPrefillCodeGen> HeuristicFilter(const std::vector<FmhaPagedKVPrefillCodeGen>& instances, 
                                                           const FmhaProblem& fmha_problem);

    /**
     * @brief Create instances from configuration specification
     * @param config Configuration with parameter ranges or fixed values
     * @param fmha_problem Problem specification for context
     * @return Vector of generated instances
     */
    std::vector<FmhaPagedKVPrefillCodeGen> CreateInstanceForConfig(const FmhaPagedKVPrefillConfig& config, const FmhaProblem& fmha_problem);

    /**
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
    std::map<std::string, FmhaPagedKVPrefillCodeGen>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:

    std::map<std::string, FmhaPagedKVPrefillCodeGen> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck