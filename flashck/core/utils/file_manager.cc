#include "flashck/core/utils/file_manager.h"

#include <chrono>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace flashck {

void FileManager::WriteFile(const std::filesystem::path& file_path, const std::string& content, bool create_dirs)
{
    if (create_dirs) {
        EnsureParentDirectories(file_path);
    }

    std::ofstream file(file_path);
    if (!file.is_open()) {
        FC_THROW(Unavailable("Unable to open file for writing: {}", file_path.string()));
    }

    file << content;
    file.close();

    if (file.fail()) {
        FC_THROW(Unavailable("Failed to write content to file: {}", file_path.string()));
    }
}

std::string FileManager::ReadFile(const std::filesystem::path& file_path)
{
    if (!FileExists(file_path)) {
        FC_THROW(Unavailable("File does not exist: {}", file_path.string()));
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        FC_THROW(Unavailable("Unable to open file for reading: {}", file_path.string()));
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    if (file.fail() && !file.eof()) {
        FC_THROW(Unavailable("Failed to read content from file: {}", file_path.string()));
    }

    return buffer.str();
}

bool FileManager::FileExists(const std::filesystem::path& file_path)
{
    return std::filesystem::exists(file_path) && std::filesystem::is_regular_file(file_path);
}

bool FileManager::CreateDirectoryIfNotExists(const std::filesystem::path& dir_path)
{
    try {
        if (std::filesystem::exists(dir_path)) {
            return std::filesystem::is_directory(dir_path);
        }
        return std::filesystem::create_directories(dir_path);
    }
    catch (const std::filesystem::filesystem_error& e) {
        return false;
    }
}

bool FileManager::DeleteFile(const std::filesystem::path& file_path)
{
    try {
        return std::filesystem::remove(file_path) || !std::filesystem::exists(file_path);
    }
    catch (const std::filesystem::filesystem_error& e) {
        return false;
    }
}

size_t FileManager::GetFileSize(const std::filesystem::path& file_path)
{
    if (!FileExists(file_path)) {
        FC_THROW(Unavailable("File does not exist: {}", file_path.string()));
    }

    try {
        return std::filesystem::file_size(file_path);
    }
    catch (const std::filesystem::filesystem_error& e) {
        FC_THROW(Unavailable("Failed to get file size for: {}", file_path.string()));
    }
}

size_t FileManager::WriteFiles(const std::vector<std::pair<std::filesystem::path, std::string>>& files,
                               bool                                                              create_dirs)
{
    size_t success_count = 0;

    for (const auto& [file_path, content] : files) {
        try {
            WriteFile(file_path, content, create_dirs);
            success_count++;
        }
        catch (const std::exception& e) {
            // Log error but continue with other files
            // You might want to add logging here
            continue;
        }
    }

    return success_count;
}

void FileManager::AppendToFile(const std::filesystem::path& file_path, const std::string& content, bool create_dirs)
{
    if (create_dirs) {
        EnsureParentDirectories(file_path);
    }

    std::ofstream file(file_path, std::ios::app);
    if (!file.is_open()) {
        FC_THROW(Unavailable("Unable to open file for appending: {}", file_path.string()));
    }

    file << content;
    file.close();

    if (file.fail()) {
        FC_THROW(Unavailable("Failed to append content to file: {}", file_path.string()));
    }
}

void FileManager::CopyFile(const std::filesystem::path& source_path,
                           const std::filesystem::path& dest_path,
                           bool                         create_dirs)
{
    if (!FileExists(source_path)) {
        FC_THROW(Unavailable("Source file does not exist: {}", source_path.string()));
    }

    if (create_dirs) {
        EnsureParentDirectories(dest_path);
    }

    try {
        std::filesystem::copy_file(source_path, dest_path, std::filesystem::copy_options::overwrite_existing);
    }
    catch (const std::filesystem::filesystem_error& e) {
        FC_THROW(
            Unavailable("Failed to copy file from {} to {}: {}", source_path.string(), dest_path.string(), e.what()));
    }
}

bool FileManager::CheckWithRetries(const std::filesystem::path& exe_path, int max_attempts, int delay_seconds)
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

std::filesystem::path FileManager::CreateTemporaryDirectory(const std::string& prefix)
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

void FileManager::EnsureParentDirectories(const std::filesystem::path& file_path)
{
    auto parent_path = file_path.parent_path();
    if (!parent_path.empty() && !std::filesystem::exists(parent_path)) {
        if (!CreateDirectoryIfNotExists(parent_path)) {
            FC_THROW(Unavailable("Failed to create parent directories for: {}", file_path.string()));
        }
    }
}

}  // namespace flashck
