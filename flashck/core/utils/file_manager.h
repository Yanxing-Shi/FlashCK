#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "flashck/core/utils/common.h"

namespace flashck {

/**
 * @class FileManager
 * @brief Utility class for file operations including writing, reading, and directory management
 */
class FileManager {
public:
    /**
     * @brief Write content to a file
     * @param file_path The path where to write the file
     * @param content The content to write
     * @param create_dirs Whether to create parent directories if they don't exist
     * @throws Unavailable if file cannot be opened or written
     */
    static void WriteFile(const std::filesystem::path& file_path, const std::string& content, bool create_dirs = true);

    /**
     * @brief Read content from a file
     * @param file_path The path of the file to read
     * @return The file content as string
     * @throws Unavailable if file cannot be opened or read
     */
    static std::string ReadFile(const std::filesystem::path& file_path);

    /**
     * @brief Check if a file exists
     * @param file_path The path to check
     * @return true if file exists, false otherwise
     */
    static bool FileExists(const std::filesystem::path& file_path);

    /**
     * @brief Create directories recursively
     * @param dir_path The directory path to create
     * @return true if directories were created or already exist, false otherwise
     */
    static bool CreateDirectoryIfNotExists(const std::filesystem::path& dir_path);

    /**
     * @brief Delete a file
     * @param file_path The path of the file to delete
     * @return true if file was deleted or didn't exist, false on error
     */
    static bool DeleteFile(const std::filesystem::path& file_path);

    /**
     * @brief Get file size in bytes
     * @param file_path The path of the file
     * @return File size in bytes
     * @throws Unavailable if file doesn't exist or cannot be accessed
     */
    static size_t GetFileSize(const std::filesystem::path& file_path);

    /**
     * @brief Write multiple files from a list of path-content pairs
     * @param files Vector of (file_path, content) pairs
     * @param create_dirs Whether to create parent directories if they don't exist
     * @return Number of files successfully written
     */
    static size_t WriteFiles(const std::vector<std::pair<std::filesystem::path, std::string>>& files,
                             bool                                                              create_dirs = true);

    /**
     * @brief Append content to an existing file
     * @param file_path The path of the file to append to
     * @param content The content to append
     * @param create_dirs Whether to create parent directories if they don't exist
     * @throws Unavailable if file cannot be opened for appending
     */
    static void
    AppendToFile(const std::filesystem::path& file_path, const std::string& content, bool create_dirs = true);

    /**
     * @brief Copy a file from source to destination
     * @param source_path Source file path
     * @param dest_path Destination file path
     * @param create_dirs Whether to create parent directories if they don't exist
     * @throws Unavailable if copy operation fails
     */
    static void
    CopyFile(const std::filesystem::path& source_path, const std::filesystem::path& dest_path, bool create_dirs = true);

    /**
     * @brief Check if a file exists with retries and exponential backoff
     * @param exe_path The path to check
     * @param max_attempts Maximum number of retry attempts (default: 3)
     * @param delay_seconds Base delay between retries in seconds (default: 5)
     * @return true if file exists, false otherwise
     * @throws std::invalid_argument for invalid parameters
     * @throws std::filesystem::filesystem_error for critical filesystem errors
     */
    static bool CheckWithRetries(const std::filesystem::path& exe_path, int max_attempts = 3, int delay_seconds = 5);

    /**
     * @brief Create a temporary directory with a random suffix
     * @param prefix The prefix for the temporary directory name
     * @return The path to the created temporary directory, empty path if creation failed
     */
    static std::filesystem::path CreateTemporaryDirectory(const std::string& prefix);

private:
    /**
     * @brief Internal helper to ensure parent directories exist
     * @param file_path The file path whose parent directories should be created
     */
    static void EnsureParentDirectories(const std::filesystem::path& file_path);
};

}  // namespace flashck
