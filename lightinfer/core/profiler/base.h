#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "lightinfer/core/utils/log.h"

namespace lightinfer {

enum class DynamicProfileStrategy {
    UNDEFINED  = 0,
    MAX        = 1,
    MIN        = 2,
    HINT       = 3,
    INTERATION = 4,
};

inline DynamicProfileStrategy StringTocProfileStrategy(const std::string& strategy_str)
{
    std::string str_upper = strategy_str;
    std::transform(str_upper.begin(), str_upper.end(), str_upper.begin(), ::toupper);

    if (str_upper == "MAX") {
        return DynamicProfileStrategy::MAX;
    }
    else if (str_upper == "MIN") {
        return DynamicProfileStrategy::MIN;
    }
    else if (str_upper == "HINT") {
        return DynamicProfileStrategy::HINT;
    }
    else {
        LOG(ERROR) << "DynamicProfileStrategy " << str_upper << " is not supported.";
        return DynamicProfileStrategy::UNDEFINED;
    }
}

inline const std::string ProfileStrategyToString(const DynamicProfileStrategy& strategy)
{
    switch (strategy) {
        case DynamicProfileStrategy::MAX:
            return "max";
        case DynamicProfileStrategy::MIN:
            return "min";
        case DynamicProfileStrategy::HINT:
            return "hint";
        default:
            LOG(ERROR) << "DataLayout " << static_cast<int>(strategy) << " is not supported.";
            return "UNDEFINED";
    }
}

/*
A data class to store profiling info.
*/
class ExecItem {
public:
    ExecItem(std::string profiling_key, std::string exec_cond, std::string algo):
        profiling_key_(profiling_key), exec_cond_(exec_cond), algo_(algo)
    {
    }

    // useful for codegen and debug
    std::string profiling_key_;
    std::string exec_cond_;
    std::string algo_;
    int64_t     split_k_ = -1;
};

enum class TensorSource {
    Input  = 0,
    Output = 1,
};

class DimInfo {
public:
    DimInfo() = default;

    explicit DimInfo(TensorSource source, int64_t tensor_idx, std::vector<int64_t> dim_idx, bool placeholder = false):
        source_(source), tensor_idx_(tensor_idx), dim_idx_(dim_idx), placeholder_(placeholder)
    {
    }

    TensorSource         source_;
    int64_t              tensor_idx_;
    std::vector<int64_t> dim_idx_;
    bool                 placeholder_;
};

}  // namespace lightinfer