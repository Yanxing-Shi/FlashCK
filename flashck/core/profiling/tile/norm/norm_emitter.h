#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "flashck/core/profiling/tile/norm/norm_codegen.h"
#include "flashck/core/profiling/tile/norm/norm_problem.h"

namespace flashck {

const std::vector<NormTileDesc> g_default_norm_tile_desc = {
    // clang-format off
    // | repeat_m | repeat_n | thread_per_block_m | thread_per_block_n | vector_n  |
    {       1,          1,        8,                   8,                   8       },
    {       1,          1,        4,                   64,                  2       },
    {       1,          1,        4,                   16,                  4       },
    {       1,          1,        4,                   64,                  1       },
    {       1,          1,        4,                   16,                  8       },
    {       1,          1,        4,                   64,                  2       }
    // clang-format on
};

/**
 * @class NormEmitter
 * @brief Manages norm operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate norm operation instances based on
 * different strategies (heuristic, autotuning, or hybrid) and manages tile
 * descriptor validation and filtering.
 */
class NormEmitter {
public:
    NormEmitter()  = default;
    ~NormEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    NormEmitter(const NormEmitter&)            = delete;
    NormEmitter& operator=(const NormEmitter&) = delete;

    /**
     * @brief Get singleton instance of NormEmitter
     * @return Pointer to the singleton instance
     */
    static NormEmitter* GetInstance()
    {
        static NormEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates if a tile descriptor is valid for the given problem
     * @param tile_desc The tile descriptor to validate
     * @param norm_problem The norm problem configuration
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const NormTileDesc& tile_desc, const NormProblem& norm_problem) const;

    /**
     * @brief Applies heuristic filtering to tile descriptors
     * @param norm_tile_desc Vector of tile descriptors to filter
     * @param norm_problem The norm problem configuration
     * @return Vector of filtered tile descriptors
     */
    std::vector<NormTileDesc> HeuristicFilter(const std::vector<NormTileDesc>& norm_tile_desc,
                                              const NormProblem&               norm_problem) const;

    /**
     * @brief Generates norm operation instances based on the problem specification
     * @param norm_problem The norm problem configuration
     * @return Map of generated norm operations organized by kind and config name
     */
    void GenerateInstances(NormProblem& norm_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const;

    // get profiling instance map for the given norm kind
    std::map<std::string, NormCodeGen>& GetInstanceMap(NormProblem norm_problem)
    {
        GenerateInstances(norm_problem);
        return instance_map_[norm_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    /**
     * @brief Creates a NormCodeGen instance from problem and tile descriptor
     * @param norm_problem The norm problem configuration
     * @param tile_desc The tile descriptor
     * @return Configured NormCodeGen instance
     */
    NormCodeGen CreateNormCodeGen(const NormProblem& norm_problem, const NormTileDesc& tile_desc) const;

    /**
     * @brief Validates mode parameter and throws if invalid
     * @param mode The mode to validate
     */
    void ValidateMode(int mode) const;

    std::map<NormKind, std::map<std::string, NormCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck