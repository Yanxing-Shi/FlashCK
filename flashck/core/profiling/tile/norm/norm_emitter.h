#pragma once

#include "flashck/core/profiling/tile/norm/norm_codegen.h"
#include "flashck/core/profiling/tile/norm/norm_problem.h"

FC_DECLARE_int32(mode);

namespace flashck {

const std::vector<NormTileDesc> g_default_norm_tile_desc = {
    // clang-format off
    // | repeat_m | repeat_n | thread_per_block_m | thread_per_block_n | vector_n  |
    {1,          1,          8,                   8,                   8},
    {1,          1,          4,                   16,                  4},
    {1,          1,          4,                   64,                  1},
    {1,          1,          4,                   16,                  8},
    {1,          1,          4,                   64,                  2}
    // clang-format on
};

class NormEmitter {
public:
    NormEmitter()  = default;
    ~NormEmitter() = default;

    bool IsValidTile(const NormTileDesc& tile_desc);

    static NormEmitter* GetInstance()
    {
        static NormEmitter instance;
        return &instance;
    }

    std::vector<NormTileDesc> HeuristicFilter(const std::vector<NormTileDesc>& norm_tile_desc);

    std::map<NormKind, std::map<std::string, NormCodeGen>> GenerateInstances(const NormProblem& norm_problem);

    int64_t GetNumInstances();

    std::map<NormKind, std::map<std::string, NormCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck