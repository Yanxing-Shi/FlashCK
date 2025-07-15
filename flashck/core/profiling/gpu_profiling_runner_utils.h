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
static const std::regex instance_name_pattern(R"(KERNEL:\s*([a-zA-Z0-9_]+))");       // Captures kernel name
static const std::regex split_k_pattern(R"(SPLIT_K:\s*([\d\.]+))");                  // split_k value
static const std::regex latency_pattern(R"(LATENCY:\s*([\d\.]+)(?:\s*ms)?)");        // Captures latency value
static const std::regex tflops_pattern(R"(TFLOPS:\s*([\d\.]+)(?:\s*Tflops)?)");      // Captures tflops value
static const std::regex bandwidth_pattern(R"(BANDWIDTH:\s*([\d\.]+)(?:\s*GB/s)?)");  // Captures bandwidth value

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
inline std::tuple<PerfResult, bool> ExtractProfilingResult(std::string output)
{
    PerfResult results;
    bool       failed = false;

    try {
        // Extract kernel configurations and times
        auto instance_name = ExtractMatches(instance_name_pattern, output);
        auto split_k       = ExtractMatches(split_k_pattern, output);
        auto latency       = ExtractMatches(latency_pattern, output);
        auto tflops        = ExtractMatches(tflops_pattern, output);
        auto bandwidth     = ExtractMatches(bandwidth_pattern, output);

        VLOG(1) << "Extracted instance name: " << (instance_name.empty() ? "N/A" : instance_name[0])
                << ", split_k: " << (split_k.empty() ? "N/A" : split_k[0])
                << ", latency: " << (latency.empty() ? "N/A" : latency[0])
                << ", tflops: " << (tflops.empty() ? "N/A" : tflops[0])
                << ", bandwidth: " << (bandwidth.empty() ? "N/A" : bandwidth[0]);

        // Check for required fields (split_k is optional)
        if (instance_name.empty() || latency.empty() || tflops.empty() || bandwidth.empty()) {
            throw std::runtime_error("Incomplete profiling data: missing one or more required fields.");
        }

        results = PerfResult{.split_k_   = split_k.empty() ? -1 : std::stoi(split_k[0]),
                             .latency_   = std::stof(latency[0]),
                             .tflops_    = std::stof(tflops[0]),
                             .bandwidth_ = std::stof(bandwidth[0])};
    }
    catch (const std::regex_error& e) {
        // Handle regex errors
        LOG(ERROR) << "Regex error in ExtractProfilingResult: " << e.what();
        failed = true;
    }
    catch (const std::invalid_argument& e) {
        // Handle invalid time values
        LOG(ERROR) << "Invalid argument in ExtractProfilingResult: " << e.what();
        failed = true;
    }
    catch (const std::exception& e) {
        // Handle other exceptions
        LOG(ERROR) << "Exception in ExtractProfilingResult: " << e.what();
        failed = true;
    }

    return std::make_tuple(results, failed);
}

}  // namespace flashck