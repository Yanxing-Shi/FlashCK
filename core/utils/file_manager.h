#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace flashck {

/**
 * @brief Utility class for file operations including writing, reading, and directory management
 */
class FileManager {
public:
    // ==============================================================================
    // Core File Operations
    // ==============================================================================

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
     * @brief Read content from a file with optional error handling
     * @param file_path The path of the file to read
     * @return Optional containing file content, or nullopt if failed
     */
    static std::optional<std::string> ReadFileOptional(const std::filesystem::path& file_path) noexcept;

    /**
     * @brief Read lines from a file into a vector
     * @param file_path The path of the file to read
     * @param skip_empty Whether to skip empty lines
     * @return Vector of lines from the file
     * @throws Unavailable if file cannot be opened or read
     */
    static std::vector<std::string> ReadLines(const std::filesystem::path& file_path, bool skip_empty = false);

    /**
     * @brief Append content to an existing file
     * @param file_path The path of the file to append to
     * @param content The content to append
     * @param create_dirs Whether to create parent directories if they don't exist
     * @throws Unavailable if file cannot be opened for appending
     */
    static void
    AppendToFile(const std::filesystem::path& file_path, const std::string& content, bool create_dirs = true);

    // ==============================================================================
    // File System Query Operations
    // ==============================================================================

    /**
     * @brief Check if a file exists and is a regular file
     * @param file_path The path to check
     * @return true if file exists and is a regular file, false otherwise
     */
    static bool FileExists(const std::filesystem::path& file_path) noexcept;

    /**
     * @brief Check if a directory exists
     * @param dir_path The path to check
     * @return true if directory exists, false otherwise
     */
    static bool DirectoryExists(const std::filesystem::path& dir_path) noexcept;

    /**
     * @brief Get file size in bytes
     * @param file_path The path of the file
     * @return File size in bytes
     * @throws Unavailable if file doesn't exist or cannot be accessed
     */
    static size_t GetFileSize(const std::filesystem::path& file_path);

    /**
     * @brief Get file modification time
     * @param file_path The path of the file
     * @return File modification time
     * @throws Unavailable if file doesn't exist or cannot be accessed
     */
    static std::filesystem::file_time_type GetFileModificationTime(const std::filesystem::path& file_path);

    // ==============================================================================
    // Directory Operations
    // ==============================================================================

    /**
     * @brief Create directories recursively
     * @param dir_path The directory path to create
     * @return true if directories were created or already exist, false otherwise
     */
    static bool CreateDirectoryIfNotExists(const std::filesystem::path& dir_path) noexcept;

    /**
     * @brief Create directories recursively with exception on failure
     * @param dir_path The directory path to create
     * @throws Unavailable if directory creation fails
     */
    static void CreateDirectories(const std::filesystem::path& dir_path);

    /**
     * @brief Remove directory and all its contents
     * @param dir_path The directory path to remove
     * @return true if directory was removed or didn't exist, false on error
     */
    static bool RemoveDirectory(const std::filesystem::path& dir_path) noexcept;

    // ==============================================================================
    // File Management Operations
    // ==============================================================================

    /**
     * @brief Delete a file
     * @param file_path The path of the file to delete
     * @return true if file was deleted or didn't exist, false on error
     */
    static bool DeleteFile(const std::filesystem::path& file_path) noexcept;

    /**
     * @brief Move/rename a file
     * @param source_path Source file path
     * @param dest_path Destination file path
     * @param create_dirs Whether to create parent directories if they don't exist
     * @throws Unavailable if move operation fails
     */
    static void
    MoveFile(const std::filesystem::path& source_path, const std::filesystem::path& dest_path, bool create_dirs = true);

    /**
     * @brief Copy a file from source to destination
     * @param source_path Source file path
     * @param dest_path Destination file path
     * @param create_dirs Whether to create parent directories if they don't exist
     * @throws Unavailable if copy operation fails
     */
    static void
    CopyFile(const std::filesystem::path& source_path, const std::filesystem::path& dest_path, bool create_dirs = true);

    // ==============================================================================
    // Batch Operations
    // ==============================================================================

    /**
     * @brief Write multiple files from a list of path-content pairs
     * @param files Vector of (file_path, content) pairs
     * @param create_dirs Whether to create parent directories if they don't exist
     * @return Number of files successfully written
     */
    static size_t WriteFiles(const std::vector<std::pair<std::filesystem::path, std::string>>& files,
                             bool                                                              create_dirs = true);

    /**
     * @brief Read multiple files into a map
     * @param file_paths Vector of file paths to read
     * @return Map of file_path -> content for successfully read files
     */
    static std::vector<std::pair<std::filesystem::path, std::string>>
    ReadFiles(const std::vector<std::filesystem::path>& file_paths);

    // ==============================================================================
    // Advanced Operations
    // ==============================================================================

    /**
     * @brief Check if a file exists with retries and exponential backoff
     * @param file_path The path to check
     * @param max_attempts Maximum number of retry attempts (default: 3)
     * @param delay_seconds Base delay between retries in seconds (default: 1)
     * @return true if file exists, false otherwise
     * @throws std::invalid_argument for invalid parameters
     * @throws std::filesystem::filesystem_error for critical filesystem errors
     */
    static bool CheckWithRetries(const std::filesystem::path& file_path, int max_attempts = 3, int delay_seconds = 1);

    /**
     * @brief Create a temporary file with unique name
     * @param prefix The prefix for the temporary file name
     * @param suffix The suffix for the temporary file name (default: ".tmp")
     * @return The path to the created temporary file
     * @throws Unavailable if temporary file creation fails
     */
    static std::filesystem::path CreateTemporaryFile(const std::string& prefix, const std::string& suffix = ".tmp");

    /**
     * @brief Create a temporary directory with a random suffix
     * @param prefix The prefix for the temporary directory name
     * @return The path to the created temporary directory
     * @throws Unavailable if temporary directory creation fails
     */
    static std::filesystem::path CreateTemporaryDirectory(const std::string& prefix);

    // ==============================================================================
    // Utility Methods
    // ==============================================================================

    /**
     * @brief Get the current working directory
     * @return Current working directory path
     * @throws Unavailable if current directory cannot be determined
     */
    static std::filesystem::path GetCurrentDirectory();

    /**
     * @brief Get the system temporary directory
     * @return System temporary directory path
     * @throws Unavailable if temp directory cannot be determined
     */
    static std::filesystem::path GetTemporaryDirectory();

    /**
     * @brief Resolve a path to its absolute form
     * @param path The path to resolve
     * @return Absolute path
     * @throws Unavailable if path cannot be resolved
     */
    static std::filesystem::path ResolvePath(const std::filesystem::path& path);

private:
    // ==============================================================================
    // Internal Helper Methods
    // ==============================================================================

    /**
     * @brief Internal helper to ensure parent directories exist
     * @param file_path The file path whose parent directories should be created
     * @throws Unavailable if parent directory creation fails
     */
    static void EnsureParentDirectories(const std::filesystem::path& file_path);

    /**
     * @brief Internal helper for optimized file writing
     * @param file_path The file path to write to
     * @param content The content to write
     * @param mode The file open mode
     * @param create_dirs Whether to create parent directories
     */
    static void WriteFileInternal(const std::filesystem::path& file_path,
                                  const std::string&           content,
                                  std::ios::openmode           mode,
                                  bool                         create_dirs);

    /**
     * @brief Generate a random string for temporary file names
     * @param length The length of the random string
     * @return Random string
     */
    static std::string GenerateRandomString(size_t length);
};

}  // namespace flashck
