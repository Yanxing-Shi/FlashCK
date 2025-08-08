#pragma once

#include <unordered_map>

#include "core/profiling/norm/norm_library.h"
#include "core/profiling/norm/rms_norm/rms_norm_codegen.h"

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
    /// @param rms_norm_problem The RMS normalization problem specification
    /// @return true if tile configuration is valid, false otherwise
    static bool IsValidTile(const RmsNormTileDesc& tile_desc, const RmsNormProblem& rms_norm_problem);

    /// @brief Validate complete RMS normalization instance
    /// @param instance The code generation instance to validate
    /// @return true if instance is valid for execution, false otherwise
    static bool IsValidInstance(const RmsNormCodeGen& instance);

    /// @brief Apply heuristic filtering to RMS normalization instances
    /// 
    /// Ranks instances based on:
    /// - Memory coalescing efficiency (vector alignment)
    /// - Thread utilization patterns
    /// - Work balance per thread
    /// - Feature dimension coverage efficiency
    /// 
    /// @param instances Input instances to filter
    /// @param rms_norm_problem The RMS normalization problem for context
    /// @return Filtered and ranked instances optimized for RMS normalization
    static std::vector<RmsNormCodeGen> HeuristicFilter(
        const std::vector<RmsNormCodeGen>& instances,
        const RmsNormProblem& rms_norm_problem);

    /// @brief Generate code instances from RMS normalization configuration
    /// @param config Configuration specifying parameter ranges
    /// @param rms_norm_problem The target RMS normalization problem
    /// @return Vector of all possible code generation instances
    std::vector<RmsNormCodeGen> CreateInstanceForConfig(
        const NormConfig& config, const RmsNormProblem& rms_norm_problem);

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
    /// @param rms_norm_problem The RMS normalization problem to generate instances for
    void GenerateInstances(RmsNormProblem& rms_norm_problem);

    /// @brief Get total number of generated valid instances
    /// @return Count of instances available for profiling
    int64_t GetNumInstances() const;

    /// @brief Clear all generated instances and reset state
    void ClearInstances();

    /// @return Map of instance name to code generation object
    const std::map<std::string, RmsNormCodeGen>& GetInstances(RmsNormProblem rms_norm_problem) {
        GenerateInstances(rms_norm_problem);
        return instance_map_;
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
    std::map<std::string, RmsNormCodeGen> instance_map_;
    
    /// @brief Total count of valid generated instances
    int64_t num_instances_ = 0;
};

}  // namespace flashck