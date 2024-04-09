#include "ater/core/profiler/build_cache.h"

#include "ater/core/profiler/build_cache_utils.h"
#include "ater/core/utils/file_utils.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/printf.h"
#include "ater/core/utils/string_utils.h"

namespace ater {

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

/*
Filesystem based build cache.

    For method docstrings, see parent class.

    Args:
        cache_dir (str): Path to store cache data below. Should be an empty, temporary directory with enough
space to hold the cache contents. Will be written to and deleted in! lru_retention_hours (int, optional):
Retention time for *unused* cache entries. Defaults to 72. cleanup_max_age_seconds (int, optional): Minimum
time between cache cleanups in seconds. After this time, a new cleanup gets triggered on next cache
retrieval. Defaults to 3600. debug (bool, optional): Whether to enable debugging cache key creation ( see
debug parameter of create_dir_hash). Defaults to True. May be left at True, as it is usually helpful and
does not hurt performance.
*/
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

void FileBasedBuildCache::CleanUp(const int lru_retention_hours)
{
    LOG(INFO) << "CACHE: Cleaning up build cache below " << cache_dir_.string() << " Folders last used more than "
              << lru_retention_hours << " hours ago will be deleted.";

    TouchFile(cache_dir_ / ".last_cleaned");
    if (std::filesystem::is_directory(cache_dir_)) {
        const auto                          start     = std::chrono::system_clock::now();
        const std::chrono::duration<double> age_limit = std::chrono::seconds(lru_retention_hours * 60 * 60 * 60);
        for (const auto& dirpath : std::filesystem::recursive_directory_iterator(cache_dir_)) {
            if (std::filesystem::is_directory(dirpath)) {
                // Get the modification time of the directory and convert it to a datetime object
                auto ftime = std::filesystem::last_write_time(dirpath);
                // Check if the directory is older than N hours
                if (start - FileToSystemTime(ftime) > age_limit) {
                    LOG(INFO) << "CACHE: Deleting " << dirpath.path().string();
                    std::filesystem::remove_all(dirpath);
                }
            }
        }
    }
}

}  // namespace ater