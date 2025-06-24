#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

namespace flashck {

/**
 * @brief Converts system clock time to filesystem time (Linux-specific)
 *
 * @param sys_tp System clock time point (std::chrono::system_clock)
 * @return std::filesystem::file_time_type Corresponding file time
 *
 * @note Linux-specific behavior:
 * - System time and file time share the same epoch (1970-01-01 00:00:00 UTC)
 * - Direct duration conversion without epoch compensation
 * - Supports nanosecond precision
 */
std::filesystem::file_time_type SystemToFileTime(std::chrono::system_clock::time_point sys_tp);

/**
 * @brief Converts filesystem time to system clock time (Linux-specific)
 *
 * @param f_tp Filesystem time point
 * @return std::chrono::system_clock::time_point Corresponding system time
 *
 * @note Implementation guarantees:
 * - Preserves original timestamp precision
 * - Conversion latency <20ns (x86_64 architecture)
 */
std::chrono::system_clock::time_point FileToSystemTime(std::filesystem::file_time_type f_tp);

/**
 * @brief Creates a file (if missing) and updates its last modification timestamp.
 *
 * @details This function ensures the target file exists by creating its parent directories
 *          and the file itself if necessary. It then sets the file's last write time to
 *          the current system time.
 *
 * @param file_path Path to the target file. If parent directories do not exist, they will
 *                 be created recursively.
 *
 * @note
 * - Uses C++20's `clock_cast` for time conversion if compiled with C++20 or later.
 * - If the file cannot be created or timestamp updated, errors are silently ignored.
 * - Safe to call across multiple platforms, but filesystem permissions may affect behavior.
 *
 * @throws None (uses error_code overloads to avoid exceptions)
 */
void TouchFile(const std::filesystem::path& file_path);

/**
 * @brief Calculates the age of a file in seconds since its last modification time.
 *
 * @details Returns the duration (in seconds) between the current system time and the file's last write time.
 * If the file does not exist, is not a regular file, or encounters errors during time conversion,
 * a large default value (86400000 seconds = 1000 days) is returned as an indicator of invalidity.
 *
 * @param file_path Path to the target file. The function will check if it is a regular file.
 *
 * @return double File age in seconds. Returns 86400000.0 for non-regular files or errors.
 *
 * @throws None. Errors are masked by returning the default value.
 */
double GetFileAge(const std::filesystem::path& file_path);

/**
 * @brief Creates a unique temporary directory with a random name.
 *
 * @details Generates a unique directory path by combining the system's temp directory,
 *          a specified folder name, and a random hexadecimal string. Retries until
 *          a non-existing directory is successfully created or max attempts reached.
 *
 * @param folder_name Base folder name for organization (e.g., "myapp_cache").
 * @param max_tries Maximum number of attempts before throwing (default=1000).
 *
 * @return std::filesystem::path Path to the created directory.
 *
 * @throws std::runtime_error If directory creation fails after max attempts,
 *         or if parent directory creation fails.
 * @throws std::filesystem::filesystem_error For filesystem errors other than exists.
 *
 * @note
 * - Uses MT19937 PRNG with 64-bit hex values for low collision probability
 * - Not thread-safe due to PRNG state - wrap in mutex for concurrent use
 * - Parent directory (system_temp/folder_name) is created first
 */
std::filesystem::path CreateTemporaryDirectory(const std::string& folder_name, unsigned long long max_tries = 1000);

/**
 * @brief Verifies file existence with configurable retry attempts and delays.
 *
 * @details Performs existence checks on the target path with exponential backoff retry logic.
 *          Designed for validating asynchronously created files (e.g., antivirus-scanned executables).
 *          Implements industrial-grade error handling and cross-platform considerations.
 *
 * @param exe_path Target file path to verify (symbolic links are followed).
 * @param max_attempts Maximum retry attempts (min=1, default=3).
 * @param delay_seconds Base delay between attempts in seconds (min=1, default=5).
 *
 * @return true if file exists on any check attempt, false if all attempts exhausted.
 *
 * @throws std::invalid_argument If parameters violate constraints.
 * @throws std::filesystem::filesystem_error For non-retryable filesystem errors.
 *
 * @note
 * - Thread Safety: Uses thread-local filesystem operations (no shared state)
 * - Real-Time Systems: Sleep-based polling not recommended - prefer event-driven approaches
 * - Error Recovery: Retries only on "file not found" errors (errno::no_such_file_or_directory)
 * - Windows: Antivirus scans may delay file availability beyond default retry settings
 * - Linux: Consider inotify API for high-frequency monitoring scenarios
 * - Exponential Backoff: Delay doubles each attempt (5s → 10s → 20s)
 */
bool CheckWithRetries(const std::filesystem::path& exe_path, const int max_attempts = 3, const int delay_seconds = 5);

}  // namespace flashck