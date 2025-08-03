#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/norm/norm_library.h"
#include "core/profiling/norm/norm_problem.h"
#include "core/profiling/norm/layer_norm/layer_norm_codegen.h"
#include "core/utils/json_config.h"

namespace flashck {

/**
 * @class LayerNormEmitter
 * @brief Manages Layer Normalization code generation and optimization
 *
 * This class provides comprehensive functionality for Layer Normalization operations:
 * - Supports three configuration types: backup (pre-validated), default (parameter ranges), user (custom)
 * - Implements three execution modes: heuristic (0), autotuning (1), hybrid (2)
 * - Generates optimized kernel instances with thread-level tiling
 * - Provides intelligent filtering for normalization optimization
 * 
 * Layer Normalization Overview:
 * - Normalizes across the feature dimension (last dimension)
 * - Computes mean and variance for each sequence element
 * - Applies scale and bias transformations
 * - Supports fused operations (add, quantization) for efficiency
 * 
 * Usage Patterns:
 * ```cpp
 * auto* emitter = LayerNormEmitter::GetInstance();
 * emitter->GenerateInstances(problem);  // Generates all valid instances
 * auto instances = emitter->HeuristicFilter(all_instances, problem);  // Optional filtering
 * ```
 */
class LayerNormEmitter {
public:
    LayerNormEmitter()  = default;
    ~LayerNormEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    LayerNormEmitter(const LayerNormEmitter&)            = delete;
    LayerNormEmitter& operator=(const LayerNormEmitter&) = delete;

    /**
     * @brief Get singleton instance of LayerNormEmitter
     * @return Pointer to the singleton instance
     */
    static LayerNormEmitter* GetInstance()
    {
        static LayerNormEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates tile configuration against problem constraints
     * @param tile_desc Tile descriptor containing thread-level parameters
     * @param norm_problem Problem specification including dimensions and data types
     * @return true if tile configuration is valid, false otherwise
     * 
     * Validation includes:
     * - Parameter positivity and alignment constraints
     * - Memory bandwidth optimization for feature dimension access
     * - Thread block size limitations
     * - Vector size compatibility with data types
     */
    bool IsValidTile(const LayerNormTileDesc& tile_desc, const NormProblem& norm_problem);

    /**
     * @brief Validates generated code instance
     * @param instance Generated kernel instance to validate
     * @return true if instance is valid and can be compiled
     */
    bool IsValidInstance(const LayerNormCodeGen& instance);

    /**
     * @brief Creates kernel instances from configuration
     * @param config Configuration with parameter ranges or single values
     * @param norm_problem Problem specification
     * @return Vector of generated kernel instances
     */
    std::vector<LayerNormCodeGen> CreateInstanceForConfig(const NormConfig& config, const NormProblem& norm_problem);

    /**
     * @brief Apply intelligent filtering to reduce search space
     * @param instances All generated instances to filter
     * @param norm_problem Problem specification for context-aware filtering
     * @return Filtered subset of instances with better performance characteristics
     * 
     * Heuristic Strategy:
     * - Prioritizes configurations with optimal memory coalescing
     * - Considers feature dimension size for efficient reduction
     * - Balances thread utilization and memory bandwidth
     * - Prefers vector sizes that align with data types
     */
    std::vector<LayerNormCodeGen> HeuristicFilter(const std::vector<LayerNormCodeGen>& instances, 
                                            const NormProblem& norm_problem);

    /**
     * @brief Main instance generation entry point supporting multiple configuration sources
     * @param norm_problem The Layer Normalization problem configuration to solve
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
    void GenerateInstances(NormProblem& norm_problem);

    /**
     * @brief Gets the total number of generated instances across all configurations
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const
    {
        return num_instances_;
    }

    /**
     * @brief Get profiling instance map for the given Layer Normalization kind
     * @param norm_problem The normalization problem configuration
     * @return Reference to the instance map for the specific operation kind
     */
    std::map<std::string, LayerNormCodeGen>& GetInstanceMap(NormProblem norm_problem)
    {
        GenerateInstances(norm_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    // Instance storage organized by normalization operation type
    std::map<std::string, LayerNormCodeGen> instance_map_;
    
    // Performance tracking
    int64_t num_instances_ = 0;
};

}  // namespace flashck
