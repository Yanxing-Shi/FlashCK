#pragma once

#include <algorithm>
#include <exception>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "flashck/core/profiling/profiling_helper.h"
#include "flashck/core/utils/common.h"

namespace flashck {

// ============================================================================
// Regular Expression Patterns for Profiling Output Parsing
// ============================================================================

/// Regex pattern to capture kernel instance name from profiling output
/// Matches format: "KERNEL: <kernel_name>"
static const std::regex instance_name_pattern(R"(KERNEL:\s*([a-zA-Z0-9_]+))");

/// Regex pattern to capture split-K configuration value
/// Matches format: "SPLIT_K: <numeric_value>"
static const std::regex split_k_pattern(R"(SPLIT_K:\s*([\d\.]+))");

/// Regex pattern to capture execution latency in milliseconds
/// Matches format: "LATENCY: <numeric_value> [ms]" (unit is optional)
static const std::regex latency_pattern(R"(LATENCY:\s*([\d\.]+)(?:\s*ms)?)");

/// Regex pattern to capture throughput in teraflops
/// Matches format: "TFLOPS: <numeric_value> [Tflops]" (unit is optional)
static const std::regex tflops_pattern(R"(TFLOPS:\s*([\d\.]+)(?:\s*Tflops)?)");

/// Regex pattern to capture memory bandwidth in GB/s
/// Matches format: "BANDWIDTH: <numeric_value> [GB/s]" (unit is optional)
static const std::regex bandwidth_pattern(R"(BANDWIDTH:\s*([\d\.]+)(?:\s*GB/s)?)");

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Extract all regex matches from input string
 * @param regex Compiled regular expression pattern
 * @param input Input string to search for matches
 * @return Vector of matched subgroups (first capture group from each match)
 *
 * This function searches the input string for all occurrences of the given
 * regex pattern and extracts the first capture group from each match.
 * Used for extracting specific values from profiling output text.
 */
inline std::vector<std::string> ExtractMatches(const std::regex& regex, std::string input)
{
    std::vector<std::string> matches;
    std::sregex_iterator     iter(input.begin(), input.end(), regex);
    std::sregex_iterator     end;

    for (; iter != end; ++iter) {
        // Only extract if there's a capture group (subgroup)
        if (iter->size() > 1) {
            matches.emplace_back(iter->str(1));
        }
    }

    VLOG(3) << "Extracted " << matches.size() << " matches from input string";
    return matches;
}

/**
 * @brief Extract performance results from profiling output string
 * @param output Raw output string from profiling execution
 * @return Tuple containing PerfResult struct and error flag
 *
 * Parses the profiling output string using regex patterns to extract:
 * - Kernel instance name (required)
 * - Split-K configuration (optional, defaults to -1)
 * - Execution latency in milliseconds (required)
 * - Throughput in teraflops (required)
 * - Memory bandwidth in GB/s (required)
 *
 * The function handles various error conditions including:
 * - Regex compilation errors
 * - Missing required fields
 * - Invalid numeric conversions
 * - General parsing exceptions
 *
 * @note Split-K value is optional and will be set to -1 if not found
 */
inline std::tuple<PerfResult, bool> ExtractProfilingResult(std::string output)
{
    PerfResult results;
    bool       failed = false;

    try {
        VLOG(2) << "Parsing profiling output (" << output.size() << " characters)";

        // Extract performance metrics using regex patterns
        auto instance_name = ExtractMatches(instance_name_pattern, output);
        auto split_k       = ExtractMatches(split_k_pattern, output);
        auto latency       = ExtractMatches(latency_pattern, output);
        auto tflops        = ExtractMatches(tflops_pattern, output);
        auto bandwidth     = ExtractMatches(bandwidth_pattern, output);

        // Log extracted values for debugging
        VLOG(1) << "Extracted profiling metrics:"
                << " instance_name=" << (instance_name.empty() ? "N/A" : instance_name[0])
                << ", split_k=" << (split_k.empty() ? "N/A" : split_k[0])
                << ", latency=" << (latency.empty() ? "N/A" : latency[0]) << "ms"
                << ", tflops=" << (tflops.empty() ? "N/A" : tflops[0])
                << ", bandwidth=" << (bandwidth.empty() ? "N/A" : bandwidth[0]) << "GB/s";

        // Validate that all required fields are present
        // Note: split_k is optional and may be absent for some kernel types
        if (instance_name.empty() || latency.empty() || tflops.empty() || bandwidth.empty()) {
            throw std::runtime_error("Incomplete profiling data: missing one or more required fields "
                                     "(instance_name, latency, tflops, or bandwidth)");
        }

        // Convert string values to numeric types and construct result
        results = PerfResult(split_k.empty() ? -1 : std::stoi(split_k[0]),  // Optional field: split_k_
                             std::stof(latency[0]),                         // Required: latency_
                             std::stof(tflops[0]),                          // Required: tflops_
                             std::stof(bandwidth[0])                        // Required: bandwidth_
        );

        VLOG(1) << "Successfully parsed profiling results";
    }
    catch (const std::regex_error& e) {
        // Handle regex compilation or execution errors
        LOG(ERROR) << "Regex error in ExtractProfilingResult: " << e.what() << " (error code: " << e.code() << ")";
        failed = true;
    }
    catch (const std::invalid_argument& e) {
        // Handle invalid numeric string conversions (stoi, stof failures)
        LOG(ERROR) << "Invalid numeric argument in ExtractProfilingResult: " << e.what()
                   << " - Check profiling output format";
        failed = true;
    }
    catch (const std::out_of_range& e) {
        // Handle numeric values that are too large for the target type
        LOG(ERROR) << "Numeric value out of range in ExtractProfilingResult: " << e.what();
        failed = true;
    }
    catch (const std::exception& e) {
        // Handle any other unexpected exceptions during parsing
        LOG(ERROR) << "Unexpected exception in ExtractProfilingResult: " << e.what()
                   << " - Please check profiling output format";
        failed = true;
    }

    return std::make_tuple(results, failed);
}

}  // namespace flashck