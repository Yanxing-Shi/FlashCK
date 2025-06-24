#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "flashck/core/utils/log.h"

namespace flashck {

// Defines strategies for dynamically profiling CK kernel performance.
// Used by the profiler engine to extract value from dynamic shape as workload.
enum class DynamicProfileStrategy {
    kMax       = 0,  // Extract the maximum value of dynamic shape as input workload.
    kMin       = 1,  // Extract the maximum value of dynamic shape as input workload.
    kHint      = 2,  // Hint the exact value of dynamic shape as input workload.
    kIteration = 3,  // Extract the value list of each dimension of dynamic shape according to step as input workload.
};

inline DynamicProfileStrategy StringToProfileStrategy(const std::string& strategy_str)
{
    if (strategy_str == "MAX") {
        return DynamicProfileStrategy::kMax;
    }
    else if (strategy_str == "MIN") {
        return DynamicProfileStrategy::kMin;
    }
    else if (strategy_str == "HINT") {
        return DynamicProfileStrategy::kHint;
    }
    else {
        LOG(ERROR) << "DynamicProfileStrategy " << strategy_str << " is not supported.";
        return DynamicProfileStrategy::UNDEFINED;
    }
}

constexpr const char* ProfileStrategyToChar(DynamicProfileStrategy strategy)
{
    switch (strategy) {
        case DynamicProfileStrategy::kMax:
            return "max";
        case DynamicProfileStrategy::kMin:
            return "min";
        case DynamicProfileStrategy::kHint:
            return "hint";
        case DynamicProfileStrategy::kIteration:
            return "iteration";
        default:
            return "undefined";
    }
}

// A data class to store profiling information.
struct ExecItem {
public:
    ExecItem(std::string profiling_key, std::string exec_cond, std::string algo):
        profiling_key_(std::move(profiling_key)), exec_cond_(std::move(exec_cond)), algo_(std::move(algo))
    {
    }

    std::string profiling_key_;
    std::string exec_cond_;
    std::string algo_;
    int64_t     split_k_ = -1;
};

// Indicates the source type of a tensor in neural network operations.
enum class TensorSource {
    kInput  = 0,  // Tensor represents operation input
    kOutput = 1,  // Tensor represents operation output
};

class DimInfo {
public:
    DimInfo() = default;

    explicit DimInfo(TensorSource source, int64_t tensor_idx, std::vector<int64_t> dim_idx, bool placeholder = false):
        source_(source), tensor_idx_(tensor_idx), dim_idx_(dim_idx), placeholder_(placeholder)
    {
    }

    // Returns the index of associated tensor.
    int64_t tensor_idx() const
    {
        return tensor_idx_;
    }

    // Returns the index of associated tensor.
    int64_t tensor_idx() const
    {
        return tensor_idx_;
    }

    // Returns dimension indices
    std::vector<int64_t> dim_idx() &&
    {
        return std::move(dim_idx_);
    }

    // Indicates whether this dimension is a placeholder.
    bool IsPlaceholder() const
    {
        return placeholder_;
    }

private:
    TensorSource         source_;
    int64_t              tensor_idx_;
    std::vector<int64_t> dim_idx_;
    bool                 placeholder_;
};

struct TensorDesc {
    DataType   dtype_;
    LayoutType layout_;
};

struct MaskEnumInfo {
    GenericAttentionMaskEnum type_;
    int64_t                  sliding_window_size_;
};

}  // namespace flashck