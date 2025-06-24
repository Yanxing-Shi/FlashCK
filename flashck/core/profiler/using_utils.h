#pragma once

namespace flashck {
// Type aliases to simplify complex nested types
using PathPair          = std::tuple<std::filesystem::path, std::filesystem::path>;
using OpProfilerList    = std::vector<PathPair>;
using GenProfilerResult = std::vector<OpProfilerList>;

using PathTuple         = std::tuple<std::filesystem::path, std::filesystem::path>;
using GenFunctionResult = std::vector<PathTuple>;

}  // namespace flashck