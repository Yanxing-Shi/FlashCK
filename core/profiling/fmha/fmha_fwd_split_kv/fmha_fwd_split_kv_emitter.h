#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/fmha/fmha_library.h"
#include "core/profiling/fmha/fmha_problem.h"
#include "core/profiling/fmha/fmha_fwd_split_kv/fmha_fwd_split_kv_codegen.h"
#include "core/utils/json_config.h"

namespace flashck {

/**
 * @class FmhaFwdSplitKVEmitter
 * @brief Manages FMHA forward split-KV pass code generation and optimization
 *
 * This class provides comprehensive functionality for FMHA forward split-KV operations:
 * - Supports three configuration types: backup (pre-validated), default (parameter ranges), user (custom)
 * - Implements three execution modes: heuristic (0), autotuning (1), hybrid (2)
 * - Generates optimized kernel instances with hierarchical tiling for split K-V processing
 * - Provides intelligent filtering for performance optimization in split scenarios
 * 
 * Architecture Overview:
 * - Block-level tiling: Divides computation across CUDA thread blocks with K-V splitting
 * - Warp-level distribution: Assigns work within each thread block for parallel K-V processing
 * - Thread-level optimization: Fine-grained vectorization and memory access for split operations
 * 
 * Split-KV Features:
 * - Enables processing of large K-V sequences by splitting across multiple kernel launches
 * - Optimizes memory bandwidth utilization for distributed K-V computation
 * - Supports efficient attention computation with reduced memory footprint per kernel
 * - Provides load balancing across GPU resources for variable sequence lengths
 * 
 * Usage Patterns:
 * ```cpp
 * auto* emitter = FmhaFwdSplitKVEmitter::GetInstance();
 * emitter->GenerateInstances(problem);  // Generates all valid split-KV instances
 * auto instances = emitter->HeuristicFilter(all_instances, problem);  // Optional filtering
 * ```
 */
class FmhaFwdSplitKVEmitter {
public:
    FmhaFwdSplitKVEmitter()  = default;
    ~FmhaFwdSplitKVEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    FmhaFwdSplitKVEmitter(const FmhaFwdSplitKVEmitter&)            = delete;
    FmhaFwdSplitKVEmitter& operator=(const FmhaFwdSplitKVEmitter&) = delete;

    /**
     * @brief Get singleton instance of FmhaFwdSplitKVEmitter
     * @return Pointer to the singleton instance
     */
    static FmhaFwdSplitKVEmitter* GetInstance()
    {
        static FmhaFwdSplitKVEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates tile configuration against problem constraints for split-KV operations
     * @param tile_desc Tile descriptor containing block/warp/thread-level parameters for split-KV
     * @param fmha_problem Problem specification including dimensions and data types
     * @return true if tile configuration is valid for split-KV operations, false otherwise
     * 
     * Split-KV Validation includes:
     * - Parameter positivity and divisibility constraints for split operations
     * - Hardware resource limitations (registers, shared memory) in split scenarios
     * - Mathematical correctness for split attention computation
     * - K-V splitting constraints and memory alignment requirements
     * - Load balancing considerations for variable split sizes
     */
    bool IsValidTile(const FmhaFwdSplitKVTileDesc& tile_desc, const FmhaProblem& fmha_problem);

    /**
     * @brief Validates generated split-KV code instance
     * @param instance Generated split-KV kernel instance to validate
     * @return true if instance is valid and can be compiled for split-KV operations
     */
    bool IsValidInstance(const FmhaFwdSplitKVCodeGen& instance);

    /**
     * @brief Creates split-KV kernel instances from configuration
     * @param config Configuration with parameter ranges or single values for split-KV
     * @param fmha_problem Problem specification
     * @return Vector of generated split-KV kernel instances
     */
    std::vector<FmhaFwdSplitKVCodeGen> CreateInstanceForConfig(const FmhaFwdSplitKVConfig& config, const FmhaProblem& fmha_problem);

    /**
     * @brief Apply intelligent filtering to reduce search space for split-KV operations
     * @param instances All generated split-KV instances to filter
     * @param fmha_problem Problem specification for context-aware filtering
     * @return Filtered subset of instances with better performance characteristics for split-KV
     * 
     * Split-KV Heuristic Strategy:
     * - Prioritizes configurations with optimal memory access patterns for split operations
     * - Considers K-V sequence splitting efficiency and load balancing
     * - Balances register usage and memory bandwidth for distributed processing
     * - Uses performance models specific to split-KV computation patterns
     * - Optimizes for inter-kernel communication overhead minimization
     */
    std::vector<FmhaFwdSplitKVCodeGen> HeuristicFilter(const std::vector<FmhaFwdSplitKVCodeGen>& instances, 
                                                      const FmhaProblem& fmha_problem);

    /**
     * @brief Main instance generation entry point for split-KV operations
     * @param fmha_problem The FMHA problem configuration to solve with split-KV approach
     * 
     * Split-KV Execution Strategy (controlled by FC_TUNING_MODE):
     * - Mode 0 (Heuristic): Apply split-KV specific filtering → select optimal subset → random sampling
     * - Mode 1 (Autotuning): Generate all valid split-KV instances → comprehensive performance search
     * - Mode 2 (Hybrid): Combine split-KV heuristics with broader search → balanced approach
     * 
     * Configuration Loading (controlled by FC_ENABLE_*_JSON flags):
     * - Backup configs: Pre-validated split-KV configurations for immediate deployment
     * - Default configs: Parameter ranges optimized for split-KV operations
     * - User configs: Custom split-KV parameter ranges for specific use cases
     */
    void GenerateInstances(FmhaProblem& fmha_problem);

    /**
     * @brief Gets the total number of generated split-KV instances across all configurations
     * @return Number of generated split-KV instances
     */
    int64_t GetNumInstances() const
    {
        return num_instances_;
    }

    /**
     * @brief Get profiling instance map for the given FMHA kind with split-KV operations
     * @param fmha_problem The FMHA problem configuration
     * @return Reference to the split-KV instance map for the specific FMHA kind
     */
    std::map<std::string, FmhaFwdSplitKVCodeGen>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_[fmha_problem.kind_];
    }

    /**
     * @brief Clears all generated split-KV instances and resets counters
     */
    void ClearInstances();

private:
    // Split-KV instance storage organized by FMHA operation type
    std::map<FmhaKind, std::map<std::string, FmhaFwdSplitKVCodeGen>> instance_map_;
    
    // Performance tracking for split-KV operations
    int64_t num_instances_ = 0;
};

}  // namespace flashck
