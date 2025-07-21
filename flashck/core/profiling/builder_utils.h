#pragma once

#include "flashck/core/utils/common.h"

FC_DECLARE_bool(FC_TIME_COMPILATION);

namespace flashck {

/**
 * @brief Conditionally wrap a command with timing measurement
 * @param cmd The command to potentially wrap with timing
 * @return The original command or wrapped with 'time' based on FC_TIME_COMPILATION flag
 *
 * When FC_TIME_COMPILATION is enabled, wraps the command with the 'time' utility
 * to measure compilation duration for performance analysis.
 */
inline std::string TimeCmd(const std::string& cmd)
{
    return FLAGS_FC_TIME_COMPILATION ? Sprintf("time -f 'exit_status=%x elapsed_sec=%e argv=\"%C\"' {}", cmd) : cmd;
}

/**
 * @brief Execute make commands with comprehensive error tracking
 * @param cmds Vector of make commands to execute sequentially [0] = clean, [1] = build
 * @param build_dir Working directory for make execution
 * @return Pair of (success_flag, output_text) indicating results
 *
 * Executes a clean command followed by a build command, capturing both
 * stdout and stderr for detailed error analysis. Returns success status
 * and complete output for failure diagnosis.
 */
inline std::pair<bool, std::string>
RunMakeCmds(const std::vector<std::string>& cmds,  // [0] = "make clean", [1] = "make run"
            const std::filesystem::path&    build_dir)
{
    VLOG(1) << "Executing make commands in directory: " << build_dir.string();
    for (size_t i = 0; i < cmds.size(); ++i) {
        VLOG(1) << "  Command[" << i << "]: " << cmds[i];
    }

    try {
        // Execute clean command first
        subprocess::check_output(SplitStrings(cmds[0], " "));

        // Execute build command with output capture
        subprocess::Popen popen(
            SplitStrings(cmds[1], " "), subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});

        auto        result     = popen.communicate();
        std::string stdout_str = result.first.buf.data();
        std::string stderr_str = result.second.buf.data();

        VLOG(2) << "Make stdout (" << stdout_str.length() << " chars): " << stdout_str;
        VLOG(2) << "Make stderr (" << stderr_str.length() << " chars): " << stderr_str;

        // Determine success based on output characteristics
        if (!stdout_str.empty() && stderr_str.empty()) {
            return {true, stdout_str};
        }
        else {
            VLOG(1) << "Make command failed in directory: " << build_dir.string() << "\nstdout: " << stdout_str
                    << "\nstderr: " << stderr_str;
            return {false, stderr_str};
        }
    }
    catch (const std::exception& e) {
        std::string error_msg = Sprintf("Exception during make execution in {}: {}", build_dir.string(), e.what());
        VLOG(1) << error_msg;
        return {false, error_msg};
    }
}

/**
 * @brief Parse failed compilation files from make output with comprehensive error detection
 * @param make_output String containing stdout/stderr output from make command
 * @return Vector of failed file names and error descriptions
 *
 * Analyzes make output to identify compilation failures, extracting:
 * - Source files that failed to compile (.cc files)
 * - Missing header files or dependencies
 * - Make target resolution failures
 * - General make execution errors
 *
 * Uses multiple heuristics to parse different types of error messages
 * and provides categorized failure information for debugging.
 */
inline std::vector<std::string> ParseFailedFiles(const std::string& make_output)
{
    std::vector<std::string> failed_files;
    std::istringstream       iss(make_output);
    std::string              line;

    while (std::getline(iss, line)) {
        // Pattern 1: Compilation errors (error:, fatal error:, make Error)
        if (line.find("error:") != std::string::npos || line.find("fatal error:") != std::string::npos
            || (line.find("make[") != std::string::npos && line.find("Error") != std::string::npos)) {

            // Extract .cc filename from error context
            size_t cc_pos = line.find(".cc");
            if (cc_pos != std::string::npos) {
                size_t path_start = line.rfind('/', cc_pos);
                if (path_start != std::string::npos) {
                    std::string filename = line.substr(path_start + 1, cc_pos - path_start + 2);
                    // Avoid duplicates
                    if (std::find(failed_files.begin(), failed_files.end(), filename) == failed_files.end()) {
                        failed_files.push_back(filename);
                        VLOG(2) << "Detected compilation failure: " << filename;
                    }
                }
            }

            // Pattern 2: Missing file/directory errors
            if (line.find("no such file or directory:") != std::string::npos) {
                size_t quote_start = line.find("'", line.find("no such file or directory:"));
                size_t quote_end   = line.find("'", quote_start + 1);
                if (quote_start != std::string::npos && quote_end != std::string::npos) {
                    std::string missing_file = line.substr(quote_start + 1, quote_end - quote_start - 1);
                    std::string error_desc   = "Missing: " + missing_file;
                    failed_files.push_back(error_desc);
                    VLOG(2) << "Detected missing file: " << missing_file;
                }
            }
        }
        // Pattern 3: Make target resolution failures
        else if (line.find("No rule to make target") != std::string::npos) {
            // Try double-quoted targets first
            size_t dquote_start = line.find("'\"");
            size_t dquote_end   = line.find("\"'", dquote_start + 2);
            if (dquote_start != std::string::npos && dquote_end != std::string::npos) {
                std::string target     = line.substr(dquote_start + 2, dquote_end - dquote_start - 2);
                std::string error_desc = "Target: " + target;
                failed_files.push_back(error_desc);
                VLOG(2) << "Detected target error (double-quoted): " << target;
            }
            // Fallback to single-quoted targets
            else {
                size_t squote_start = line.find("'");
                size_t squote_end   = line.find("'", squote_start + 1);
                if (squote_start != std::string::npos && squote_end != std::string::npos) {
                    std::string target     = line.substr(squote_start + 1, squote_end - squote_start - 1);
                    std::string error_desc = "Target: " + target;
                    failed_files.push_back(error_desc);
                    VLOG(2) << "Detected target error (single-quoted): " << target;
                }
            }
        }
        // Pattern 4: General make execution failures
        else if (line.find("make: ***") != std::string::npos && line.find("Error") != std::string::npos) {
            // Extract the failed target from make error message
            size_t colon_pos   = line.find(": ");
            size_t bracket_pos = line.find("]");
            if (colon_pos != std::string::npos && bracket_pos != std::string::npos && bracket_pos > colon_pos) {
                std::string target     = line.substr(colon_pos + 2, bracket_pos - colon_pos - 2);
                std::string error_desc = "Failed: " + target;
                failed_files.push_back(error_desc);
                VLOG(2) << "Detected make execution error: " << target;
            }
        }
    }

    if (!failed_files.empty()) {
        VLOG(1) << "Parsed " << failed_files.size() << " compilation failures from make output";
    }
    return failed_files;
}

/**
 * @brief Format compilation statistics for consistent logging output
 * @param successful Number of successful compilations
 * @param total Total number of compilation attempts
 * @param phase_name Optional phase identifier (e.g., "Tuning", "Running")
 * @return Formatted string with compilation statistics and success rate
 *
 * Creates a standardized format for compilation statistics reporting:
 * "[Phase] compilation statistics - Success: X/Y (Z.Z%)"
 * Used for consistent logging across tuning and running phases.
 */
inline std::string FormatCompilationStats(size_t successful, size_t total, const std::string& phase_name = "")
{
    double success_rate = total > 0 ? static_cast<double>(successful) / total * 100.0 : 0.0;

    std::ostringstream oss;
    if (!phase_name.empty()) {
        oss << phase_name << " ";
    }
    oss << "compilation statistics - Success: " << successful << "/" << total << " (" << std::fixed
        << std::setprecision(2) << success_rate << "%)";

    return oss.str();
}
}  // namespace flashck