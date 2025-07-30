
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/tile/fmha/fmha_library.h"
#include "core/profiling/tile/fmha/fmha_problem.h"

#include "core/profiling/tile/fmha/fmha_fwd_split_kv/fmha_fwd_split_kv_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_split_kv/fmha_fwd_split_kv_backup_config.h"

namespace flashck {

/**
 * @class FmhaFwdSplitKVEmitter
 * @brief Manages FMHA operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate FMHA operation instances based on
 * different strategies (heuristic, autotuning, or hybrid) and manages tile
 * descriptor validation and filtering. Interface is designed to be consistent with GemmEmitter.
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

    bool IsValidTile(const FmhaFwdSplitKVTileDesc& tile_desc, const FmhaProblem& fmha_problem);

    bool IsValidInstance(const FmhaFwdSplitKVCodeGen& instance);

    std::vector<FmhaFwdSplitKVCodeGen> CreateInstanceForConfig(const FmhaFwdSplitKVConfig& config, const FmhaProblem& fmha_problem);

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
    std::map<std::string, FmhaFwdSplitKVCodeGen>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_[fmha_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    
    std::map<FmhaKind, std::map<std::string, FmhaFwdSplitKVCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck