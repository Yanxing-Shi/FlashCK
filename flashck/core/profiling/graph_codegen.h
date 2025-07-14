#pragma once

#include <filesystem>
#include <vector>

#include "flashck/core/graph/node.h"
#include "flashck/core/profiling/profiling_strategy.h"
#include "flashck/core/utils/common.h"

namespace flashck {

using PathPair          = std::tuple<std::filesystem::path, std::filesystem::path>;
using OpProfilerList    = std::vector<PathPair>;
using GenProfilerResult = std::vector<OpProfilerList>;

using PathTuple         = std::tuple<std::filesystem::path, std::filesystem::path>;
using GenFunctionResult = std::vector<PathTuple>;

// Generates profiling file paths for operations supporting dynamic profiling.

class GraphCodeGen {
public:
    GenProfilerResult CodeGenForTuning(const std::vector<Operation*>& model_ops,
                                       const ProfilingStrategy&       strategy = ProfilingStrategy::kMax);

    void CodeGenAndProfiling(const std::vector<Operation*>& model_ops,
                             const std::string&             context_name,
                             const ProfilingStrategy&       strategy    = ProfilingStrategy::kMax,
                             const std::string&             folder_name = "kernel_profile");
};

}  // namespace flashck