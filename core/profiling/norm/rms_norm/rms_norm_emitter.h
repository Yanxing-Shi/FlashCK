#pragma once

#include <unordered_map>

#include "core/profiling/norm/norm_library.h"
#include "core/profiling/norm/norm_problem.h"
#include "core/profiling/norm/rms_norm/rms_norm_codegen.h"
#include "core/utils/json_config.h"

namespace flashck {
class RmsNormEmitter {
public:
    /// @brief Get singleton instance of RMS normalization emitter
    /// @return Reference to the singleton RmsNormEmitter instance
    static RmsNormEmitter& GetInstance() {
        static RmsNormEmitter instance;
        return instance;
    }

    /// @brief Validate RMS normalization tile configuration against problem constraints
    /// @param tile_desc The tile descriptor to validate
    /// @param norm_problem The RMS normalization problem specification
    /// @return true if tile configuration is valid, false otherwise
    static bool IsValidTile(const RmsNormTileDesc& tile_desc, const NormProblem& norm_problem);

    /// @brief Validate complete RMS normalization instance
    /// @param instance The code generation instance to validate
    /// @return true if instance is valid for execution, false otherwise
    static bool IsValidInstance(const NormCodeGen& instance);

    /// @brief Apply heuristic filtering to RMS normalization instances
    /// 
    /// Ranks instances based on:
    /// - Memory coalescing efficiency (vector alignment)
    /// - Thread utilization patterns
    /// - Work balance per thread
    /// - Feature dimension coverage efficiency
    /// 
    /// @param instances Input instances to filter
    /// @param norm_problem The RMS normalization problem for context
    /// @return Filtered and ranked instances optimized for RMS normalization
    static std::vector<NormCodeGen> HeuristicFilter(
        const std::vector<NormCodeGen>& instances,
        const NormProblem& norm_problem);

    /// @brief Generate code instances from RMS normalization configuration
    /// @param config Configuration specifying parameter ranges
    /// @param norm_problem The target RMS normalization problem
    /// @return Vector of all possible code generation instances
    std::vector<NormCodeGen> CreateInstanceForConfig(
        const RmsNormConfig& config, const NormProblem& norm_problem);

    /// @brief Generate and store RMS normalization instances based on configuration and tuning mode
    /// 
    /// Loads configurations from JSON files based on enabled flags:
    /// - backup_config.json: Pre-validated single configurations
    /// - default_config.json: Default parameter ranges
    /// - user_config.json: Custom parameter ranges
    /// 
    /// Applies mode-specific processing:
    /// - Mode 0 (heuristic): Fast execution with filtered selection
    /// - Mode 1 (autotuning): Comprehensive search with all instances
    /// - Mode 2 (hybrid): Balanced approach with intelligent filtering
    /// 
    /// @param norm_problem The RMS normalization problem to generate instances for
    void GenerateInstances(NormProblem& norm_problem);

    /// @brief Get total number of generated valid instances
    /// @return Count of instances available for profiling
    int64_t GetNumInstances() const;

    /// @brief Clear all generated instances and reset state
    void ClearInstances();

    /// @brief Get instances for specific normalization kind
    /// @param kind The normalization operation kind
    /// @return Map of instance name to code generation object
    const std::unordered_map<std::string, NormCodeGen>& GetInstances(NormKindEnum kind) const {
        static const std::unordered_map<std::string, NormCodeGen> empty_map;
        auto it = instance_map_.find(kind);
        return (it != instance_map_.end()) ? it->second : empty_map;
    }

    // Disable copy/move operations for singleton
    RmsNormEmitter(const RmsNormEmitter&) = delete;
    RmsNormEmitter& operator=(const RmsNormEmitter&) = delete;
    RmsNormEmitter(RmsNormEmitter&&) = delete;
    RmsNormEmitter& operator=(RmsNormEmitter&&) = delete;

private:
    /// @brief Private constructor for singleton pattern
    RmsNormEmitter() = default;

    /// @brief Storage for generated instances organized by normalization kind
    std::unordered_map<NormKindEnum, std::unordered_map<std::string, NormCodeGen>> instance_map_;
    
    /// @brief Total count of valid generated instances
    int64_t num_instances_ = 0;
};

}  // namespace flashck