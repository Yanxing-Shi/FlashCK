
#pragma once

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/profiling/attention/fmha_library.h"
#include "core/profiling/attention/fmha_fwd_batch_prefill_problem.h"
#include "core/utils/json_config.h"

#include "core/profiling/attention/fmha_fwd_batch_prefill/fmha_fwd_batch_prefill_codegen.h"

namespace flashck {

/**
 * @class FmhaFwdBatchPrefillEmitter
 * @brief Manages FMHA batch prefill operation code generation and tile selection
 *
 * This class provides functionality to generate FMHA operation instances based on
 * three strategies: heuristic (mode 0), autotuning (mode 1), or hybrid (mode 2).
 * Supports three config types: default, backup, and user configurations.
 * Interface is designed to be consistent with GemmEmitter.
 */
class FmhaFwdBatchPrefillEmitter {
public:
    FmhaFwdBatchPrefillEmitter()  = default;
    ~FmhaFwdBatchPrefillEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    FmhaFwdBatchPrefillEmitter(const FmhaFwdBatchPrefillEmitter&)            = delete;
    FmhaFwdBatchPrefillEmitter& operator=(const FmhaFwdBatchPrefillEmitter&) = delete;

    /**
     * @brief Get singleton instance of FmhaFwdBatchPrefillEmitter
     * @return Pointer to the singleton instance
     */
    static FmhaFwdBatchPrefillEmitter* GetInstance()
    {
        static FmhaFwdBatchPrefillEmitter instance;
        return &instance;
    }

    /**
     * @brief Validate tile descriptor against problem constraints
     * @param tile_desc Tile configuration to validate
     * @param fmha_fwd_batch_prefill_problem Problem specification for validation context
     * @return true if tile is valid, false otherwise
     */
    bool IsValidTile(const FmhaFwdBatchPrefillTileDesc& tile_desc, const FmhaFwdBatchPrefillProblem& fmha_fwd_batch_prefill_problem);

    /**
     * @brief Validate complete FMHA instance configuration
     * @param instance Complete instance to validate
     * @return true if instance is valid, false otherwise
     */
    bool IsValidInstance(const FmhaFwdBatchPrefillCodeGen& instance);

    /**
     * @brief Generate all possible instances from a configuration
     * @param config Configuration containing parameter ranges
     * @param fmha_fwd_batch_prefill_problem Problem specification
     * @return Vector of generated instances (Cartesian product of parameters)
     */
    std::vector<FmhaFwdBatchPrefillCodeGen> CreateInstanceForConfig(const FmhaFwdBatchPrefillConfig& config, 
                                                                const FmhaFwdBatchPrefillProblem& fmha_fwd_batch_prefill_problem);

    /**
     * @brief Apply heuristic filtering to reduce instance count
     * @param instances Input instance list
     * @param fmha_fwd_batch_prefill_problem Problem specification for filtering context
     * @return Filtered instance list optimized for the given problem
     */
    std::vector<FmhaFwdBatchPrefillCodeGen> HeuristicFilter(const std::vector<FmhaFwdBatchPrefillCodeGen>& instances,
                                                        const FmhaFwdBatchPrefillProblem& fmha_fwd_batch_prefill_problem) const;

    /**
     * @brief Generates FMHA operation instances based on the problem specification
     * 
     * Supports three modes:
     * - Mode 0 (Heuristic): Apply heuristic filtering + random selection
     * - Mode 1 (Autotuning): Use all valid instances for comprehensive search
     * - Mode 2 (Hybrid): Combine heuristic and autotuning strategies
     * 
     * Config types loaded based on flags:
     * - FC_ENABLE_BACKUP_JSON: Load backup_config.json (pre-validated single configs)
     * - FC_ENABLE_DEFAULT_JSON: Load default_config.json (parameter ranges for tuning)
     * - FC_ENABLE_USER_JSON: Load user_config.json (custom user configurations)
     * 
     * @param fmha_fwd_batch_prefill_problem The FMHA problem configuration
     */
    void GenerateInstances(FmhaFwdBatchPrefillProblem& fmha_fwd_batch_prefill_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances across all FMHA kinds
     */
    int64_t GetNumInstances() const { return num_instances_; }

    /**
     * @brief Get profiling instance map for the given FMHA kind
     * @param fmha_fwd_batch_prefill_problem The FMHA problem configuration
     * @return Reference to the instance map for the specific FMHA kind
     */
    std::map<std::string, FmhaFwdBatchPrefillCodeGen>& GetInstanceMap(FmhaFwdBatchPrefillProblem fmha_fwd_batch_prefill_problem)
    {
        GenerateInstances(fmha_fwd_batch_prefill_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    /// Instance storage: {instance_name -> CodeGen}
    std::map<std::string, FmhaFwdBatchPrefillCodeGen> instance_map_;

    /// Total number of generated instances across all kinds
    int64_t num_instances_ = 0;
    
    /// Random number generator for heuristic mode selection
    mutable std::mt19937 rng_{std::random_device{}()};
};

}  // namespace flashck