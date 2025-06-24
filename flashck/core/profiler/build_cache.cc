#include "flashck/core/profiler/build_cache.h"

#include "flashck/core/profiler/build_cache_utils.h"
#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/log.h"
#include "flashck/core/utils/printf.h"
#include "flashck/core/utils/string_utils.h"

namespace flashck {

NoBuildCache::NoBuildCache()
{
    LOG(INFO) << "Build cache disabled.";
}

std::tuple<bool, std::string>
NoBuildCache::RetrieveBuildCache(const std::vector<std::string>&          cmds,
                                 const std::filesystem::path&             build_dir,
                                 const std::function<bool(std::string&)>& from_sources_filter_func)
{
    return std::make_tuple(false, "");
}

bool NoBuildCache::StoreBuildCache(const std::vector<std::string>&          cmds,
                                   const std::filesystem::path&             build_dir,
                                   const std::string&                       cache_key,
                                   const std::function<bool(std::string&)>& filter_func)
{
    return false;
}

FileBasedBuildCache::FileBasedBuildCache(const std::filesystem::path& cache_dir,
                                         const int                    lru_retention_hours,
                                         const int                    cleanup_max_age_seconds,
                                         bool                         debug):
    cache_dir_(cache_dir),
    lru_retention_hours_(lru_retention_hours),
    cleanup_max_age_seconds_(cleanup_max_age_seconds),
    debug_(debug)
{
    LOG(INFO) << "Using file-based build cache, cache directory " << cache_dir_.string();
}

std::tuple<bool, std::string>
FileBasedBuildCache::RetrieveBuildCache(const std::vector<std::string>&          cmds,
                                        const std::filesystem::path&             build_dir,
                                        const std::function<bool(std::string&)>& from_sources_filter_func)
{
    if (ShouldSkipBuildCache()) {
        LOG(WARNING) << "CACHE: Skipped build cache for " << build_dir.string();
        return std::make_tuple(false, "");
    }

    MaybeCleanUp(lru_retention_hours_, cleanup_max_age_seconds_);

    const std::string     dir_hash      = CreateDirHash(cmds, build_dir, from_sources_filter_func, debug_);
    std::filesystem::path key_cache_dir = cache_dir_ / dir_hash;
    if (std::filesystem::exists(key_cache_dir)) {
        LOG(INFO) << "CACHE: Using cached build results for " << build_dir.string();

        std::vector<std::filesystem::path> copy_files;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(build_dir)) {
            if (!std::filesystem::is_directory(entry)) {
                copy_files.emplace_back(std::filesystem::relative(entry, build_dir));
            }
        }

        for (const auto& file_path : copy_files) {
            std::filesystem::path target_path   = build_dir / file_path;
            std::filesystem::path target_parent = target_path.parent_path();
            std::filesystem::path src_path      = key_cache_dir / file_path;
            if (target_parent != build_dir) {
                std::filesystem::create_directories(target_parent);
            }

            std::filesystem::copy(src_path, target_path, std::filesystem::copy_options::copy_symlinks);
            LOG(INFO) << "CACHE: retrieved " << file_path.string();
        }

        // make sure the last modified timestamp is updated, so we can
        // evict cache directories which are too old using a separate script
        std::filesystem::last_write_time(key_cache_dir, std::filesystem::file_time_type::clock::now());
        return std::make_tuple(true, dir_hash);
    }
    else {
        LOG(WARNING) << "CACHE: No build cache found for " << build_dir.string();
        return std::make_tuple(false, dir_hash);
    }
}

bool FileBasedBuildCache::StoreBuildCache(const std::vector<std::string>&          cmds,
                                          const std::filesystem::path&             build_dir,
                                          const std::string&                       cache_key,
                                          const std::function<bool(std::string&)>& filter_func)
{
    std::filesystem::path key_cache_dir = cache_dir_ / cache_key;

    // generate a temporary directory name, random hexadecimal string
    std::string random_str = Sprintf("{}", std::rand());

    // We create a temporary directory first, so we can do an
    // atomic update later to prevent race conditions
    // in a distributed / parallel build setting
    std::filesystem::path temp_cache_dir = key_cache_dir / Sprintf("{}.tmp", random_str);
    bool                  if_create      = std::filesystem::create_directories(temp_cache_dir);
    if (if_create) {
        LOG(INFO) << "CACHE: Created cache directory " << temp_cache_dir.string();
    }
    else {
        LOG(WARNING) << "CACHE: Failed to create cache directory " << temp_cache_dir.string();
        return false;
    }

    std::vector<std::filesystem::path> copy_files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(build_dir)) {
        if (!std::filesystem::is_directory(entry)) {
            copy_files.emplace_back(std::filesystem::relative(entry, build_dir));
        }
    }

    for (const auto& file_path : copy_files) {
        std::filesystem::path src_path      = build_dir / file_path;
        std::filesystem::path target_path   = temp_cache_dir / file_path;
        std::filesystem::path target_parent = target_path.parent_path();
        if (target_parent != temp_cache_dir) {
            std::filesystem::create_directories(target_parent);
        }

        std::filesystem::copy(src_path, target_path, std::filesystem::copy_options::copy_symlinks);
        LOG(INFO) << "CACHE: stored " << file_path.string() << " into " << key_cache_dir.string();
        try {
            std::filesystem::rename(temp_cache_dir, key_cache_dir);
        }
        catch (const std::exception& e) {
            LOG(ERROR) << "CACHE: update race conflict and " << key_cache_dir.string()
                       << " already exists. (Note: No error! This can be expected to happen occasionally.";
            std::filesystem::remove_all(temp_cache_dir);
            return false;
        }
    }
    return true;
}

void FileBasedBuildCache::MaybeCleanUp(const int lru_retention_hours, const int cleanup_max_age_seconds)
{
    int last_cleaned_seconds = GetFileAge(cache_dir_ / ".last_cleaned");
    if (last_cleaned_seconds > cleanup_max_age_seconds) {
        CleanUp(lru_retention_hours);
    }
}

/**
 * @brief Deletes cache directories older than a specified retention period.
 *
 * @details Recursively traverses the cache directory and removes subdirectories whose
 *          last modification time exceeds the retention threshold. Logs actions via
 *          the application's logging system.
 *
 * @param lru_retention_hours Retention period in hours. Directories older than this
 *                           will be deleted. Must be a positive integer.
 *
 * @note
 * - Uses `recursive_directory_iterator` to traverse subdirectories.
 * - Skips directories with permission errors during traversal.
 * - Time calculations assume system_clock and filesystem clock are synchronized,
 *   which may not hold true on all platforms.
 * - Logs errors but does not propagate exceptions.
 *
 * @throws std::filesystem::filesystem_error If critical filesystem operations fail.
 */
void FileBasedBuildCache::CleanUp(const int lru_retention_hours)
{
    // Validate input
    if (lru_retention_hours <= 0) {
        LOG(ERROR) << "Invalid retention hours: " << lru_retention_hours;
        return;
    }

    LOG(INFO) << "CACHE: Cleaning up build cache below " << cache_dir_.string() << ". Folders last used more than "
              << lru_retention_hours << " hours ago will be deleted.";

    // Update last cleaned timestamp
    TouchFile(cache_dir_ / ".last_cleaned");

    // Early exit if cache directory is missing
    if (!std::filesystem::exists(cache_dir_)) {
        LOG(WARNING) << "CACHE: Cache directory does not exist: " << cache_dir_.string();
        return;
    }

    const auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(lru_retention_hours);

    // Configure directory iterator
    auto dir_iter = std::filesystem::recursive_directory_iterator(
        cache_dir_,
        std::filesystem::directory_options::skip_permission_denied
            | std::filesystem::directory_options::follow_directory_symlink);

    // Traverse and clean
    for (const auto& entry : dir_iter) {
        if (!entry.is_directory())
            continue;

        try {
            const auto ftime = std::filesystem::last_write_time(entry);
#if __cplusplus >= 202002L
            const auto sys_time = std::chrono::clock_cast<std::chrono::system_clock>(ftime);
#else
            const auto sys_time = FileToSystemTime(ftime);  // Assume custom implementation
#endif

            if (sys_time < cutoff_time) {
                LOG(INFO) << "CACHE: Deleting " << entry.path().string();
                std::filesystem::remove_all(entry.path());
            }
        }
        catch (const std::filesystem::filesystem_error& e) {
            LOG(ERROR) << "CACHE: Failed to process " << entry.path().string() << ": " << e.what();
        }
    }
}

}  // namespace flashck