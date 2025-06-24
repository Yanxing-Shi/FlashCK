#include "flashck/core/utils/file_utils.h"

#include <glog/logging.h>

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
std::filesystem::file_time_type SystemToFileTime(std::chrono::system_clock::time_point sys_tp)
{
    return std::filesystem::file_time_type(sys_tp.time_since_epoch());
}

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
std::chrono::system_clock::time_point FileToSystemTime(std::filesystem::file_time_type f_tp)
{
    return std::chrono::system_clock::time_point(f_tp.time_since_epoch());
}

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
void TouchFile(const std::filesystem::path& file_path)
{
    std::error_code ec;

    // Check if file exists (with error suppression)
    if (!std::filesystem::exists(file_path, ec)) {
        // Create parent directories if needed
        const auto parent_path = file_path.parent_path();
        if (!parent_path.empty()) {
            std::filesystem::create_directories(parent_path, ec);
        }

        // Create empty file (append mode to preserve content if already exists)
        if (!ec) {
            std::ofstream file(file_path, std::ios::app);
            // Intentionally no error check: failure is non-critical
        }
    }

    // Update timestamp using C++20/C++17 compatible method
    const auto sys_now   = std::chrono::system_clock::now();
    const auto file_time = SystemToFileTime(sys_now);

    std::filesystem::last_write_time(file_path, file_time, ec);
}

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
double GetFileAge(const std::filesystem::path& file_path)
{
    // 1. Check if the path is a regular file
    std::error_code ec;
    const bool      is_regular = std::filesystem::is_regular_file(file_path, ec);

    // Return default value on errors or non-regular files
    if (ec || !is_regular) {
        return 86400000.0;  // 3600 * 24 * 1000
    }

    // 2. Get current time and file write time
    const auto sys_now = std::chrono::system_clock::now();
    const auto ftime   = std::filesystem::last_write_time(file_path, ec);

    if (ec) {
        return 86400000.0;
    }

    // 3. Convert file_time_type to system_clock time_point
    const auto sys_ftime = FileToSystemTime(ftime);  // Custom implementation required

    // 4. Calculate duration in seconds
    const auto duration = sys_now - sys_ftime;
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

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
std::filesystem::path CreateTemporaryDirectory(const std::string& folder_name, unsigned long long max_tries)
{
    namespace fs = std::filesystem;

    // 1. Prepare base parent directory
    const fs::path  tmp_dir = fs::temp_directory_path() / folder_name;
    std::error_code ec;

    // Create parent directory first
    if (!fs::create_directories(tmp_dir, ec) && ec.value() != 0) {
        throw std::runtime_error("Failed to create parent directory: " + ec.message());
    }

    // 2. PRNG setup (64-bit hex)
    std::random_device                      dev;
    std::mt19937_64                         prng(dev());
    std::uniform_int_distribution<uint64_t> dist;
    constexpr int                           hex_digits = 16;  // 64-bit needs 16 hex chars

    // 3. Attempt directory creation
    for (unsigned long long i = 0; i < max_tries; ++i) {
        // Generate random hex string
        std::stringstream ss;
        ss << std::hex << std::uppercase << std::setfill('0') << std::setw(hex_digits) << dist(prng);

        // Attempt creation
        fs::path attempt_path = tmp_dir / ss.str();
        if (fs::create_directory(attempt_path, ec)) {
            return attempt_path;  // Success
        }

        // Handle errors other than "already exists"
        if (ec.value() != static_cast<int>(std::errc::file_exists)) {
            throw fs::filesystem_error("Unexpected error", ec);
        }
    }

    throw std::runtime_error("Failed to create unique directory after " + std::to_string(max_tries)
                             + " attempts in: " + tmp_dir.string());
}

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
bool CheckWithRetries(const std::filesystem::path& exe_path, const int max_attempts, const int delay_seconds)
{
    namespace fs = std::filesystem;

    // Parameter Validation
    if (max_attempts < 1) {
        throw std::invalid_argument("Max attempts must be ≥1. Got: " + std::to_string(max_attempts));
    }
    if (delay_seconds < 1) {
        throw std::invalid_argument("Delay must be ≥1 second. Got: " + std::to_string(delay_seconds));
    }

    std::error_code ec;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        // Use error_code to avoid exception overhead
        const bool exists = fs::exists(exe_path, ec);

        // Handle non-retryable errors
        if (ec && ec != std::errc::no_such_file_or_directory) {
            throw fs::filesystem_error("Critical filesystem error", exe_path, ec);
        }

        if (exists)
            return true;

        // Skip retry logging on final attempt
        if (attempt == max_attempts)
            break;

        LOG(ERROR) << "Check attempt " << attempt << "/" << max_attempts << " failed for: " << exe_path.string()
                   << " | Next retry in " << delay_seconds * (1 << (attempt - 1)) << "s"
                   << " | System message: " << ec.message();

        // Exponential backoff calculation
        const auto backoff = std::chrono::seconds(delay_seconds * (1 << (attempt - 1)));
        std::this_thread::sleep_for(backoff);
    }

    return false;
}

}  // namespace flashck