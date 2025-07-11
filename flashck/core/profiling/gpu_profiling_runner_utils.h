#pragma once

#include <algorithm>
#include <exception>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "flashck/core/utils/common.h"

#include "flashck/core/profiling/codegen_utils.h"

namespace flashck {

// Constants for profiler configuration
static constexpr int g_profiler_run_max_attempts        = 3;
static constexpr int g_profiler_run_retry_delay_seconds = 10000;

// Regular expressions for parsing kernel and time values
static const std::regex instance_name_pattern(R"(KERNEL:([a-zA-Z0-9_]+))");  // Captures kernel name
static const std::regex split_k_pattern(R"(SPLIT_K:([\d\.]+))");             // split_k value
static const std::regex latency_pattern(R"(TIME:([\d\.]+))");                // Captures latency value
static const std::regex tflops_pattern(R"(TFLOPS:([\d\.]+))");               // Captures tflops value
static const std::regex bandwidth_pattern(R"(BANDWIDTH:([\d\.]+))");         // Captures bandwidth value

inline std::vector<std::string> ExtractMatches(const std::regex& regex, std::string input)
{
    std::vector<std::string> matches;
    std::sregex_iterator     iter(input.begin(), input.end(), regex);
    std::sregex_iterator     end;

    for (; iter != end; ++iter) {
        if (iter->size() > 1) {
            matches.emplace_back(iter->str(1));
        }
    }
    return matches;
}

// Extracts profiling results from a given output string.
inline std::tuple<std::vector<PerfResult>, bool> ExtractProfileResult(std::string                  output,
                                                                      const std::set<std::string>& return_kernels = {})
{
    std::vector<PerfResult> results;
    bool                    failed = false;

    try {
        // Extract kernel configurations and times
        auto instance_name = ExtractMatches(instance_name_pattern, output);
        auto split_k       = ExtractMatches(split_k_pattern, output);
        auto latency       = ExtractMatches(latency_pattern, output);
        auto tflops        = ExtractMatches(tflops_pattern, output);
        auto bandwidth     = ExtractMatches(bandwidth_pattern, output);

        // Ensure we have matching pairs of kernel configs and latency
        if (instance_name.size() != latency.size() || instance_name.size() != tflops.size()
            || instance_name.size() != bandwidth.size()) {
            FC_THROW(Unavailable("Mismatched number of kernel configurations and time values."));
        }

        // Populate results
        for (size_t i = 0; i < instance_name.size(); ++i) {
            // Filter kernels if return_kernels is specified
            if (return_kernels.empty() || return_kernels.count(std::string(instance_name[i])) > 0) {
                results.push_back(PerfResult{.split_k_   = split_k.empty() ? -1 : std::stoi(split_k[i]),
                                             .latency_   = std::stof(latency[i]),
                                             .tflops_    = std::stof(tflops[i]),
                                             .bandwidth_ = std::stof(bandwidth[i])});
            }
        }
    }
    catch (const std::regex_error& e) {
        // Handle regex errors
        failed = true;
    }
    catch (const std::invalid_argument& e) {
        // Handle invalid time values
        failed = true;
    }
    catch (const std::exception& e) {
        // Handle other exceptions
        failed = true;
    }

    return std::make_tuple(results, failed);
}

}  // namespace flashck