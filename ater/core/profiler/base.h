#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "ater/core/utils/log.h"

namespace ater {

enum class DynamicProfileStrategy {
    UNDEFINED = 0,
    MAX       = 1,
    MIN       = 2,
    HINT      = 3,
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

    void SetGemmExecCondRange(std::map<std::string, std::vector<int>> range)
    {
        m_lower_bound_ = range["M"][0];
        m_upper_bound_ = range["M"][1];
        n_lower_bound_ = range["N"][0];
        n_upper_bound_ = range["N"][1];
        k_lower_bound_ = range["K"][0];
        k_upper_bound_ = range["K"][1];
    }

    std::string GetGemmAlgo(const int m, const int n, const int k)
    {
        if (m >= m_lower_bound_ && m <= m_upper_bound_ && n >= n_lower_bound_ && n <= n_upper_bound_
            && k >= k_lower_bound_ && k <= k_upper_bound_) {
            return algo_;
        }
        return "";
    }

    // useful for codegen and debug
    std::string profiling_key_;
    std::string exec_cond_;
    std::string algo_;

    // useful for runtime
    int m_lower_bound_;
    int m_upper_bound_;
    int n_lower_bound_;
    int n_upper_bound_;
    int k_lower_bound_;
    int k_upper_bound_;
};

/*
Class to record dimension info.

source:
            Source.INPUT or Source.OUTPUT
        tensor_idx:
            Depending on source, extract info from inputs[tensor_idx] or outputs[tensor_idx]
        dim_idx:
            Extract shape from inputs/outputs[tensor_idx][dim_idx]
        placeholder:
            If True, the diminfo might not be accurate in compile time, just a placeholder to be filled afterwards
            This is useful to handle issue such as broadcasting which B might not be exact.
*/
enum class TensorSource {
    Input  = 0,
    Output = 1,
};

class DimInfo {
public:
    DimInfo() = default;

    explicit DimInfo(TensorSource source, int tensor_idx, std::vector<int> dim_idx, bool placeholder = false):
        source_(source), tensor_idx_(tensor_idx), dim_idx_(dim_idx), placeholder_(placeholder)
    {
    }

    TensorSource     source_;
    int              tensor_idx_;
    std::vector<int> dim_idx_;
    bool             placeholder_;
};

}  // namespace ater