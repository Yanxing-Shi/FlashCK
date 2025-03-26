#pragma once

#include <algorithm>
#include <exception>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "lightinfer/core/utils/log.h"

#include "lightinfer/core/utils/named_tuple_utils.h"
#include "lightinfer/core/utils/string_utils.h"

namespace lightinfer {

static const std::regex g_kernel_pattern(R"(KERNEL:([a-zA-Z0-9_]+))");
static const std::regex g_time_pattern(R"(TIME:([\d\.]+))");

static constexpr int g_profiler_run_max_attempts        = 3;
static constexpr int g_profiler_run_retry_delay_seconds = 10000;

// ProfileResult = namedtuple("ProfileResult", "kernel_config duration workspace")
// Object to store profiling result
// https://github.com/johnjohnlin/namedtuple
struct ProfileResult {
    std::string kernel_config;
    float       duration;

    // copy constructor
    ProfileResult(const std::string& kernel_config, float duration): kernel_config(kernel_config), duration(duration) {}

    MAKE_NAMEDTUPLE(kernel_config, duration)
};

inline const std::tuple<std::vector<ProfileResult>, bool>
ExtractProfileResult(const std::string& output, const std::set<std::string>& return_kernels = {})
{

    bool                       failed = false;
    std::vector<ProfileResult> result;

    auto regrex_func = [&](const std::regex& regrex, const std::string& output) {
        std::vector<std::string> runtimes;
        std::smatch              m;
        std::string              s = output;
        while (std::regex_search(s, m, regrex)) {
            runtimes.emplace_back(m[0]);
            s = m.suffix().str();
        }
        return runtimes;
    };

    try {
        auto kernel_config = SliceString(JoinToString(regrex_func(g_kernel_pattern, output)), ":");
        auto times         = std::atof(SliceString(JoinToString(regrex_func(g_time_pattern, output)), ":").c_str());
        result.emplace_back(kernel_config, times);
    }
    catch (std::exception& e) {
        result.emplace_back("", 0);
        failed = true;
    }

    return std::make_tuple(result, failed);
}
}  // namespace lightinfer