#pragma once

#include <algorithm>
#include <exception>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "ater/core/utils/log.h"

#include "ater/core/utils/named_tuple_utils.h"
#include "ater/core/utils/string_utils.h"

namespace ater {

static const std::regex g_profiler_runntime_pattern(R"(KERNEL:([a-zA-Z0-9_]+),TIME:([\d\.]+),WS:([\d]+))");

static const std::regex g_kernel_pattern(R"(KERNEL:([a-zA-Z0-9_]+))");
static const std::regex g_time_pattern(R"(TIME:([\d\.]+))");
static const std::regex g_workspace_pattern(R"(WS:([\d]+))");

static constexpr int g_profiler_run_max_attempts        = 3;
static constexpr int g_profiler_run_retry_delay_seconds = 10000;

// ProfileResult = namedtuple("ProfileResult", "kernel_config duration workspace")
// Object to store profiling result
// https://github.com/johnjohnlin/namedtuple
struct ProfileResult {
    std::string kernel_config;
    float       duration;
    int         workspace;
    MAKE_NAMEDTUPLE(kernel_config, duration, workspace)
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
        // std::vector<std::string> runtimes;
        // std::smatch              m;
        // std::string              s = output;
        // while (std::regex_search(s, m, g_profiler_runntime_pattern)) {
        //     runtimes.emplace_back(m[0]);
        //     s = m.suffix().str();
        // }
        // if (runtimes.size() > 0) {
        //     LOG(INFO) << "all runtimes (unsorted):" << JoinToString(runtimes);
        //     // format - KERNEL:xx,TIME:x.xx,WS:xx
        //     if (return_kernels.size()) {
        //         for (auto& runtime : runtimes) {
        //             if (return_kernels.find(runtimes[0]) != return_kernels.end())
        //                 result.emplace_back(&runtime[0], static_cast<float>(runtime[1]),
        //                 static_cast<int>(runtime[2]));
        //         }
        //     }
        //     else {
        //         // std::string best_runtime = *std::min_element(
        //         //     runtimes.begin(), runtimes.end(), [&](const std::string& a, const std::string& b) {
        //         //         return static_cast<float>(a[1]) < static_cast<float>(b[1]);
        //         //     });

        //         VLOG(1) << "best duration:" << best_runtime[1];
        //         VLOG(1) << "best workspace:" << best_runtime[2];
        //         result.emplace_back(
        //             &best_runtime[0], static_cast<float>(best_runtime[1]), static_cast<int>(best_runtime[2]));
        //     }
        // }

        // auto kernel_config = SliceString(regrex_func(g_kernel_pattern, output), "KERNEL:");
        // auto times         = std::atof(SliceString(regrex_func(g_time_pattern, output), "TIME:").c_str());
        // auto workspaces    = std::atoi(SliceString(regrex_func(g_workspace_pattern, output), "WS:").c_str());
        auto kernel_config = SliceString(JoinToString(regrex_func(g_kernel_pattern, output)), ":");
        auto times         = std::atof(SliceString(JoinToString(regrex_func(g_time_pattern, output)), ":").c_str());
        auto workspaces = std::atoi(SliceString(JoinToString(regrex_func(g_workspace_pattern, output)), ":").c_str());
        result.emplace_back(kernel_config, times, workspaces);
    }
    catch (std::exception& e) {
        result.emplace_back("", 0, 0);
        failed = true;
    }

    return std::make_tuple(result, failed);
}
}  // namespace ater