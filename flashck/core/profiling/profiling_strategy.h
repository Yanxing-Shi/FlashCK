#pragma once

namespace flashck {

// Defines strategies for dynamically profiling CK kernel performance.
// Used by the profiler engine to extract value from dynamic shape as workload.
enum class ProfilingStrategy {
    kMax       = 0,  // Extract the maximum value of dynamic shape as input workload.
    kMin       = 1,  // Extract the maximum value of dynamic shape as input workload.
    kHint      = 2,  // Hint the exact value of dynamic shape as input workload.
    kIteration = 3,  // Extract the value list of each dimension of dynamic shape according to step as input workload.
};

template<ProfilingStrategy strategy>
struct ProfilingStrategyTraits;

template<>
struct ProfilingStrategyTraits<ProfilingStrategy::kMax> {
    static constexpr const char* name = "max";
};

template<>
struct ProfilingStrategyTraits<ProfilingStrategy::kMin> {
    static constexpr const char* name = "min";
};

template<>
struct ProfilingStrategyTraits<ProfilingStrategy::kHint> {
    static constexpr const char* name = "hint";
};

template<>
struct ProfilingStrategyTraits<ProfilingStrategy::kIteration> {
    static constexpr const char* name = "iteration";
};

}  // namespace flashck