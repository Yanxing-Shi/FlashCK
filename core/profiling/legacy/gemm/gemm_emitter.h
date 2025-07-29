#pragma once

#include "core/profiling/legacy/gemm/gemm_codegen.h"
#include "core/profiling/legacy/gemm/gemm_library.h"
#include "core/profiling/legacy/gemm/gemm_problem.h"
#include "core/profiling/legacy/gemm/gemm_emitter_helper.h"

#include "core/utils/common.h"

namespace flashck {

namespace legacy{
// GemmSpecialization name mapping for backward compatibility
const std::map<std::string, GemmSpecialization> g_gemm_spec_names = {{"", GemmSpecialization::Default},
                                                                     {"M", GemmSpecialization::MPadding},
                                                                     {"N", GemmSpecialization::NPadding},
                                                                     {"K", GemmSpecialization::KPadding},
                                                                     {"MN", GemmSpecialization::MNPadding},
                                                                     {"MK", GemmSpecialization::MKPadding},
                                                                     {"NK", GemmSpecialization::NKPadding},
                                                                     {"MNK", GemmSpecialization::MNKPadding}};

/**
 * @class GemmEmitter
 * @brief Manages GEMM operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate GEMM operation instances based on
 * different strategies (heuristic, autotuning, or hybrid) and manages tile
 * descriptor validation and filtering.
 */
class GemmEmitter {
public:
    GemmEmitter()  = default;
    ~GemmEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    GemmEmitter(const GemmEmitter&)            = delete;
    GemmEmitter& operator=(const GemmEmitter&) = delete;

    /**
     * @brief Get singleton instance of GemmEmitter
     * @return Pointer to the singleton instance
     */
    static GemmEmitter* GetInstance()
    {
        static GemmEmitter instance;
        return &instance;
    }

    bool IsValidTile(const GemmTileDesc& tile_desc, const GemmProblem& gemm_problem) const;

    bool IsValidBlockTransfer(const GemmTileDesc& tile_desc, const BlockTransferDesc& block_transfer_desc) const;

    bool IsValidCBlockTransfer(const GemmTileDesc& tile_desc, const CBlockTransferDesc& c_block_transfer_desc) const;

    bool IsValidInstance(const GemmCodegen& gemm_instance, const GemmProblem& gemm_problem) const;

    /**
     * @brief Generates GEMM operation instances based on the problem specification
     * @param gemm_problem The GEMM problem configuration
     */
    void GenerateInstances(GemmProblem& gemm_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const;

    /**
     * @brief Get profiling instance map for the given GEMM kind
     * @param gemm_problem The GEMM problem configuration
     * @return Reference to the instance map for the specific GEMM kind
     */
    std::map<std::string, GemmCodegen>& GetInstanceMap(GemmProblem gemm_problem)
    {
        GenerateInstances(gemm_problem);
        return instance_map_[gemm_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    
    std::map<GemmKind, std::map<std::string, GemmCodegen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

} // namespace legacy
}  // namespace flashck