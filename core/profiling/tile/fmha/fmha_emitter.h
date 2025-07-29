
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/tile/fmha/fmha_fwd_append_kv_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_codegen.h"
#include "core/profiling/tile/fmha/fmha_batch_prefill_codegen.h"
#include "core/profiling/tile/fmha/fmha_paged_kv_prefill_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_split_kv_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_split_kv_combine_codegen.h"

#include "core/profiling/tile/fmha/fmha_library.h"
#include "core/profiling/tile/fmha/fmha_problem.h"
#include "core/profiling/tile/fmha/fmha_emitter_helper.h"

namespace flashck {

/**
 * @class FmhaEmitter
 * @brief Manages FMHA operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate FMHA operation instances based on
 * different strategies (heuristic, autotuning, or hybrid) and manages tile
 * descriptor validation and filtering. Interface is designed to be consistent with GemmEmitter.
 */
class FmhaEmitter {
public:
    FmhaEmitter()  = default;
    ~FmhaEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    FmhaEmitter(const FmhaEmitter&)            = delete;
    FmhaEmitter& operator=(const FmhaEmitter&) = delete;

    /**
     * @brief Get singleton instance of FmhaEmitter
     * @return Pointer to the singleton instance
     */
    static FmhaEmitter* GetInstance()
    {
        static FmhaEmitter instance;
        return &instance;
    }

    bool IsValidTile(const FmhaFwdTileDesc& tile_desc, const FmhaProblem& fmha_problem);

    bool IsValidInstance(const FmhaFwdCodeGen& instance);


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
    std::map<std::string, FmhaFwdCodeGen>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_[fmha_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    
    std::map<FmhaKind, std::map<std::string, FmhaFwdCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck