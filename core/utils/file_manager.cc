#include "core/utils/file_manager.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "core/utils/enforce.h"

namespace flashck {

// ==============================================================================
// Core File Operations
// ==============================================================================

void FileManager::WriteFile(const std::filesystem::path& file_path, const std::string& content, bool create_dirs)
{
    WriteFileInternal(file_path, content, std::ios::out | std::ios::trunc, create_dirs);
}

std::string FileManager::ReadFile(const std::filesystem::path& file_path)
{
    if (!FileExists(file_path)) {
        FC_THROW(Unavailable("File does not exist: {}", file_path.string()));
    }

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        FC_THROW(Unavailable("Unable to open file for reading: {}", file_path.string()));
    }

    // Get file size for efficient string allocation
    file.seekg(0, std::ios::end);
    const auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string content;
    content.reserve(static_cast<size_t>(file_size));

    // Read file content efficiently
    content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

    if (file.fail() && !file.eof()) {
        FC_THROW(Unavailable("Failed to read content from file: {}", file_path.string()));
    }

    return content;
}

std::optional<std::string> FileManager::ReadFileOptional(const std::filesystem::path& file_path) noexcept
{
    try {
        return ReadFile(file_path);
    }
    catch (...) {
        return std::nullopt;
    }
}

std::vector<std::string> FileManager::ReadLines(const std::filesystem::path& file_path, bool skip_empty)
{
    if (!FileExists(file_path)) {
        FC_THROW(Unavailable("File does not exist: {}", file_path.string()));
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        FC_THROW(Unavailable("Unable to open file for reading: {}", file_path.string()));
    }

    std::vector<std::string> lines;
    std::string              line;

    while (std::getline(file, line)) {
        if (!skip_empty || !line.empty()) {
            lines.push_back(std::move(line));
        }
    }

    if (file.fail() && !file.eof()) {
        FC_THROW(Unavailable("Failed to read lines from file: {}", file_path.string()));
    }

    return lines;
}

void FileManager::AppendToFile(const std::filesystem::path& file_path, const std::string& content, bool create_dirs)
{
    WriteFileInternal(file_path, content, std::ios::out | std::ios::app, create_dirs);
}

// ==============================================================================
// File System Query Operations
// ==============================================================================

bool FileManager::FileExists(const std::filesystem::path& file_path) noexcept
{
    std::error_code ec;
    return std::filesystem::exists(file_path, ec) && std::filesystem::is_regular_file(file_path, ec) && !ec;
}

bool FileManager::DirectoryExists(const std::filesystem::path& dir_path) noexcept
{
    std::error_code ec;
    return std::filesystem::exists(dir_path, ec) && std::filesystem::is_directory(dir_path, ec) && !ec;
}

size_t FileManager::GetFileSize(const std::filesystem::path& file_path)
{
    if (!FileExists(file_path)) {
        FC_THROW(Unavailable("File does not exist: {}", file_path.string()));
    }

    std::error_code ec;
    const auto      size = std::filesystem::file_size(file_path, ec);
    if (ec) {
        FC_THROW(Unavailable("Failed to get file size for: {} - {}", file_path.string(), ec.message()));
    }

    return size;
}

std::filesystem::file_time_type FileManager::GetFileModificationTime(const std::filesystem::path& file_path)
{
    if (!FileExists(file_path)) {
        FC_THROW(Unavailable("File does not exist: {}", file_path.string()));
    }

    std::error_code ec;
    const auto      time = std::filesystem::last_write_time(file_path, ec);
    if (ec) {
        FC_THROW(Unavailable("Failed to get file modification time for: {} - {}", file_path.string(), ec.message()));
    }

    return time;
}

// ==============================================================================
// Directory Operations
// ==============================================================================

bool FileManager::CreateDirectoryIfNotExists(const std::filesystem::path& dir_path) noexcept
{
    std::error_code ec;

    if (std::filesystem::exists(dir_path, ec)) {
        return std::filesystem::is_directory(dir_path, ec) && !ec;
    }

    const bool created = std::filesystem::create_directories(dir_path, ec);
    return created && !ec;
}

void FileManager::CreateDirectories(const std::filesystem::path& dir_path)
{
    if (!CreateDirectoryIfNotExists(dir_path)) {
        FC_THROW(Unavailable("Failed to create directory: {}", dir_path.string()));
    }
}

bool FileManager::RemoveDirectory(const std::filesystem::path& dir_path) noexcept
{
    std::error_code ec;
    const auto      removed = std::filesystem::remove_all(dir_path, ec);
    return (removed > 0 || !std::filesystem::exists(dir_path, ec)) && !ec;
}

// ==============================================================================
// File Management Operations
// ==============================================================================

bool FileManager::DeleteFile(const std::filesystem::path& file_path) noexcept
{
    std::error_code ec;
    const bool      removed = std::filesystem::remove(file_path, ec);
    return (removed || !std::filesystem::exists(file_path, ec)) && !ec;
}

void FileManager::MoveFile(const std::filesystem::path& source_path,
                           const std::filesystem::path& dest_path,
                           bool                         create_dirs)
{
    if (!FileExists(source_path)) {
        FC_THROW(Unavailable("Source file does not exist: {}", source_path.string()));
    }

    if (create_dirs) {
        EnsureParentDirectories(dest_path);
    }

    std::error_code ec;
    std::filesystem::rename(source_path, dest_path, ec);
    if (ec) {
        FC_THROW(Unavailable(
            "Failed to move file from {} to {}: {}", source_path.string(), dest_path.string(), ec.message()));
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

    std::error_code ec;
    std::filesystem::copy_file(source_path, dest_path, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
        FC_THROW(Unavailable(
            "Failed to copy file from {} to {}: {}", source_path.string(), dest_path.string(), ec.message()));
    }
}

// ==============================================================================
// Batch Operations
// ==============================================================================

size_t FileManager::WriteFiles(const std::vector<std::pair<std::filesystem::path, std::string>>& files,
                               bool                                                              create_dirs)
{
    size_t success_count = 0;

    for (const auto& [file_path, content] : files) {
        try {
            WriteFile(file_path, content, create_dirs);
            ++success_count;
        }
        catch (const std::exception& e) {
            // Log error but continue with other files
            // You might want to add logging here
            continue;
        }
    }

    return success_count;
}

std::vector<std::pair<std::filesystem::path, std::string>>
FileManager::ReadFiles(const std::vector<std::filesystem::path>& file_paths)
{
    std::vector<std::pair<std::filesystem::path, std::string>> results;
    results.reserve(file_paths.size());

    for (const auto& file_path : file_paths) {
        try {
            auto content = ReadFile(file_path);
            results.emplace_back(file_path, std::move(content));
        }
        catch (const std::exception& e) {
            // Skip files that can't be read
            continue;
        }
    }

    return results;
}

// ==============================================================================
// Advanced Operations
// ==============================================================================

bool FileManager::CheckWithRetries(const std::filesystem::path& file_path, int max_attempts, int delay_seconds)
{
    // Parameter validation
    if (max_attempts < 1) {
        throw std::invalid_argument("Max attempts must be >= 1. Got: " + std::to_string(max_attempts));
    }
    if (delay_seconds < 1) {
        throw std::invalid_argument("Delay must be >= 1 second. Got: " + std::to_string(delay_seconds));
    }

    std::error_code ec;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        // Use error_code to avoid exception overhead
        const bool exists = std::filesystem::exists(file_path, ec);

        // Handle non-retryable errors
        if (ec && ec != std::errc::no_such_file_or_directory) {
            throw std::filesystem::filesystem_error("Critical filesystem error", file_path, ec);
        }

        if (exists) {
            return true;
        }

        // Skip sleep on final attempt
        if (attempt < max_attempts) {
            // Exponential backoff with jitter
            const auto backoff = std::chrono::seconds(delay_seconds * (1 << (attempt - 1)));
            std::this_thread::sleep_for(backoff);
        }
    }

    return false;
}

std::filesystem::path FileManager::CreateTemporaryFile(const std::string& prefix, const std::string& suffix)
{
    auto temp_dir = GetTemporaryDirectory();

    for (int attempt = 0; attempt < 10; ++attempt) {
        auto random_name = prefix + GenerateRandomString(8) + suffix;
        auto temp_path   = temp_dir / random_name;

        // Try to create the file exclusively
        std::ofstream file(temp_path, std::ios::out | std::ios::binary);
        if (file.is_open()) {
            file.close();
            return temp_path;
        }
    }

    FC_THROW(Unavailable("Failed to create temporary file with prefix: {}", prefix));
}

std::filesystem::path FileManager::CreateTemporaryDirectory(const std::string& prefix)
{
    auto temp_dir = GetTemporaryDirectory();

    for (int attempt = 0; attempt < 10; ++attempt) {
        auto random_name = prefix + GenerateRandomString(8);
        auto temp_path   = temp_dir / random_name;

        std::error_code ec;
        if (std::filesystem::create_directory(temp_path, ec) && !ec) {
            return temp_path;
        }
    }

    FC_THROW(Unavailable("Failed to create temporary directory with prefix: {}", prefix));
}

// ==============================================================================
// Utility Methods
// ==============================================================================

std::filesystem::path FileManager::GetCurrentDirectory()
{
    std::error_code ec;
    auto            path = std::filesystem::current_path(ec);
    if (ec) {
        FC_THROW(Unavailable("Failed to get current directory: {}", ec.message()));
    }
    return path;
}

std::filesystem::path FileManager::GetTemporaryDirectory()
{
    std::error_code ec;
    auto            path = std::filesystem::temp_directory_path(ec);
    if (ec) {
        FC_THROW(Unavailable("Failed to get temporary directory: {}", ec.message()));
    }
    return path;
}

std::filesystem::path FileManager::ResolvePath(const std::filesystem::path& path)
{
    std::error_code ec;
    auto            resolved = std::filesystem::absolute(path, ec);
    if (ec) {
        FC_THROW(Unavailable("Failed to resolve path {}: {}", path.string(), ec.message()));
    }
    return resolved;
}

// ==============================================================================
// Internal Helper Methods
// ==============================================================================

void FileManager::EnsureParentDirectories(const std::filesystem::path& file_path)
{
    auto parent_path = file_path.parent_path();
    if (!parent_path.empty() && !DirectoryExists(parent_path)) {
        CreateDirectories(parent_path);
    }
}

void FileManager::WriteFileInternal(const std::filesystem::path& file_path,
                                    const std::string&           content,
                                    std::ios::openmode           mode,
                                    bool                         create_dirs)
{
    if (create_dirs) {
        EnsureParentDirectories(file_path);
    }

    std::ofstream file(file_path, mode);
    if (!file.is_open()) {
        FC_THROW(Unavailable("Unable to open file for writing: {}", file_path.string()));
    }

    file.write(content.data(), content.size());

    if (file.fail()) {
        FC_THROW(Unavailable("Failed to write content to file: {}", file_path.string()));
    }
}

std::string FileManager::GenerateRandomString(size_t length)
{
    static const char                charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_int_distribution<size_t> dis(0, sizeof(charset) - 2);

    std::string result;
    result.reserve(length);

    for (size_t i = 0; i < length; ++i) {
        result += charset[dis(gen)];
    }

    return result;
}

}  // namespace flashck
