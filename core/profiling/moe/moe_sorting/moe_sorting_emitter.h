#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_problem.h"
#include "core/profiling/moe/moe_sorting/moe_sorting_codegen.h"
#include "core/utils/json_config.h"

namespace flashck {

/**
 * @class MoeSortingEmitter
 * @brief Manages MoE sorting code generation and optimization
 *
 * This class provides comprehensive functionality for MoE sorting operations:
 * - Supports three configuration types: backup (pre-validated), default (parameter ranges), user (custom)
 * - Implements three execution modes: heuristic (0), autotuning (1), hybrid (2)
 * - Generates optimized kernel instances for expert token sorting
 * - Provides intelligent filtering for load balancing optimization
 * 
 * MoE Sorting Overview:
 * - Sorts tokens by expert assignment for batched processing
 * - Optimizes memory access patterns for expert routing
 * - Implements efficient radix sort for expert indices
 * - Supports both stable and unstable sorting strategies
 * 
 * Usage Patterns:
 * ```cpp
 * auto* emitter = MoeSortingEmitter::GetInstance();
 * emitter->GenerateInstances(problem);  // Generates all valid instances
 * auto instances = emitter->HeuristicFilter(all_instances, problem);  // Optional filtering
 * ```
 */
class MoeSortingEmitter {
public:
    MoeSortingEmitter()  = default;
    ~MoeSortingEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    MoeSortingEmitter(const MoeSortingEmitter&)            = delete;
    MoeSortingEmitter& operator=(const MoeSortingEmitter&) = delete;

    /**
     * @brief Get singleton instance of MoeSortingEmitter
     * @return Pointer to the singleton instance
     */
    static MoeSortingEmitter* GetInstance()
    {
        static MoeSortingEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates sorting configuration against problem constraints
     * @param instance Generated sorting instance to validate
     * @return true if configuration is valid, false otherwise
     * 
     * Validation includes:
     * - Parameter positivity and alignment constraints
     * - Memory bandwidth optimization for sorting operations
     * - Thread block size limitations
     * - Expert tile size compatibility with problem dimensions
     */
    bool IsValidInstance(const MoeSortingCodeGen& instance);

    /**
     * @brief Creates kernel instances from configuration
     * @param config Configuration with parameter ranges or single values
     * @param moe_problem Problem specification
     * @return Vector of generated kernel instances
     */
    std::vector<MoeSortingCodeGen> CreateInstanceForConfig(const MoeSortingConfig& config, const MoeProblem& moe_problem);

    /**
     * @brief Apply intelligent filtering to reduce search space
     * @param instances All generated instances to filter
     * @param moe_problem Problem specification for context-aware filtering
     * @return Filtered subset of instances with better performance characteristics
     * 
     * Heuristic Strategy:
     * - Prioritizes configurations with optimal memory access patterns
     * - Considers expert distribution and load balancing
     * - Balances sorting throughput and memory bandwidth
     * - Prefers configurations that minimize expert imbalance
     */
    std::vector<MoeSortingCodeGen> HeuristicFilter(const std::vector<MoeSortingCodeGen>& instances, 
                                                  const MoeProblem& moe_problem);

    /**
     * @brief Main instance generation entry point supporting multiple configuration sources
     * @param moe_problem The MoE sorting problem configuration to solve
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
     * @brief Get profiling instance map for the given MoE sorting kind
     * @param moe_problem The MoE sorting problem configuration
     * @return Reference to the instance map for the specific operation kind
     */
    std::map<std::string, MoeSortingCodeGen>& GetInstanceMap(MoeProblem moe_problem)
    {
        GenerateInstances(moe_problem);
        return instance_map_[moe_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    // Instance storage organized by MoE operation type
    std::map<MoeSortingKind, std::map<std::string, MoeSortingCodeGen>> instance_map_;
    
    // Performance tracking
    int64_t num_instances_ = 0;
};

}  // namespace flashck
