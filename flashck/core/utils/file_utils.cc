#include "flashck/core/utils/file_utils.h"

#include <chrono>
#include <fstream>
#include <random>
#include <stdexcept>
#include <thread>

#include <glog/logging.h>

namespace flashck {

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

std::filesystem::path CreateTemporaryDirectory(const std::string& prefix)
{
    auto         temp_base = std::filesystem::temp_directory_path();
    std::mt19937 gen(std::random_device{}());

    for (int i = 0; i < 3; ++i) {
        std::string suffix(8, '\0');
        std::generate_n(suffix.begin(), 8, [&] { return "0123456789abcdefghijklmnopqrstuvwxyz"[gen() % 36]; });

        auto path = temp_base / (prefix + suffix);
        if (std::filesystem::create_directory(path))
            return path;
    }
    return {};
}

}  // namespace flashck