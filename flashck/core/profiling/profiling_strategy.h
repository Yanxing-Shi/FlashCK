#pragma once

#include <string>
namespace flashck {

// Defines strategies for dynamically profiling CK kernel performance.
// Used by the profiler engine to extract value from dynamic shape as workload.
enum class ProfilingStrategy {
    kMax       = 0,  // Extract the maximum value of dynamic shape as input workload.
    kMin       = 1,  // Extract the maximum value of dynamic shape as input workload.
    kHint      = 2,  // Hint the exact value of dynamic shape as input workload.
    kIteration = 3,  // Extract the value list of each dimension of dynamic shape according to step as input workload.
};

inline std::string ProfilingStrategyToString(ProfilingStrategy strategy)
{
    switch (strategy) {
        case ProfilingStrategy::kMax:
            return "Max";
        case ProfilingStrategy::kMin:
            return "Min";
        case ProfilingStrategy::kHint:
            return "Hint";
        case ProfilingStrategy::kIteration:
            return "Iteration";
        default:
            return "Unknown";
    }
}

}  // namespace flashck