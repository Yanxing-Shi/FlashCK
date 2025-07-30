#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/profiling/tile/gemm/gemm_codegen.h"
#include "core/profiling/tile/gemm/gemm_problem.h"

#include "core/profiling/tile/gemm/gemm_backup_config.h"

namespace flashck {

namespace tile {

static const std::vector<std::tuple<int, int, int>> g_tile_gemm_allowed_warp_combinations = {
        {1, 4, 1}, {2, 2, 1}, {4, 1, 1}
};

// Set of unsupported combinations: (pipeline, epilogue, scheduler)
const std::set<std::tuple<PipelineVersionEnum, EpilogueEnum, PipelineSchedulerEnum>> g_tile_gemm_unsupported_combinations = {
    {GetPipelineVersionEnumFromString("compv3"), GetEpilogueEnumFromString("cshuffle"), GetPipelineSchedulerEnumFromString("interwave")},
    {GetPipelineVersionEnumFromString("compv3"), GetEpilogueEnumFromString("default"),  GetPipelineSchedulerEnumFromString("interwave")},
    {GetPipelineVersionEnumFromString("compv4"), GetEpilogueEnumFromString("cshuffle"), GetPipelineSchedulerEnumFromString("interwave")},
    {GetPipelineVersionEnumFromString("compv4"), GetEpilogueEnumFromString("default"),  GetPipelineSchedulerEnumFromString("interwave")}
};

// Supported warp tile combinations by arch and dtype
static const std::map<std::string, std::map<std::string, std::vector<std::array<int64_t, 3>>>> g_tile_gemm_warp_tile_supported_combinations = {
    {"gfx90a", {
        {"fp16_fp16_fp16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"bf16_bf16_bf16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"fp8_fp8_fp16",   {{32,32,16},{32,32,32}}},
        {"bf8_bf8_fp16",   {{32,32,16},{32,32,32}}}
    }},
    {"gfx942", {
        {"fp16_fp16_fp16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"bf16_bf16_bf16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"fp8_fp8_fp16",   {{32,32,16},{32,32,32},{16,16,32},{16,16,64}}},
        {"bf8_bf8_fp16",   {{32,32,16},{32,32,32},{16,16,64},{16,16,32}}},
        {"int8_int8_int32", {{16,16,32},{32,32,16}}}
    }},
    {"gfx950", {
        {"fp16_fp16_fp16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"bf16_bf16_bf16", {{32,32,8},{16,16,16},{32,32,16},{16,16,32},{4,64,16},{64,4,16}}},
        {"fp8_fp8_fp16",   {{32,32,16},{32,32,32},{16,16,32},{16,16,64},{16,16,128},{32,32,64}}},
        {"bf8_bf8_fp16",   {{32,32,16},{32,32,32},{16,16,64},{16,16,32},{16,16,128},{32,32,64}}}
    }}
};



/**
 * @class GemmEmitter
 * @brief Manages norm operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate norm operation instances based on
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

    /**
     * @brief Validates if a tile descriptor is valid for the given problem
     * @param tile_desc The tile descriptor to validate
     * @param norm_problem The norm problem configuration
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const GemmTileDesc& tile_desc, const GemmProblem& gemm_problem) const;

    /**
     * @brief Applies heuristic filtering to tile descriptors
     * @param gemm_tile_desc Vector of tile descriptors to filter
     * @param gemm_problem The gemm problem configuration
     * @return Vector of filtered tile descriptors
     */
    // std::vector<GemmTileDesc> HeuristicFilter(const std::vector<GemmTileDesc>& gemm_tile_desc,
    //                                           const GemmProblem&               gemm_problem) const;

    bool IsValidCombination(const PipelineVersionEnum& pipeline, const EpilogueEnum& epilogue, const PipelineSchedulerEnum& scheduler);

    bool IsValidInstance(const GemmCodeGen& instance);

    std::vector<GemmCodeGen> CreateInstanceForConfig(const flashck::TileGemmConfig& config, const GemmProblem& gemm_problem);

    /**
     * @brief Generates gemm operation instances based on the problem specification
     * @param gemm_problem The gemm problem configuration
     * @return Map of generated gemm operations organized by kind and config name
     */
    void GenerateInstances(GemmProblem& gemm_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const;

    // get profiling instance map for the given norm kind
    std::map<std::string, GemmCodeGen>& GetInstanceMap(GemmProblem gemm_problem)
    {
        GenerateInstances(gemm_problem);
        return instance_map_[gemm_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:

    std::map<GemmKind, std::map<std::string, GemmCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

} // namespace tile

}  // namespace flashck