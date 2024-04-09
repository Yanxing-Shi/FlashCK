#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <tuple>

#include "ater/core/profiler/build_cache_utils.h"
#include "ater/core/utils/flags.h"

ATER_DECLARE_string(ATER_BUILD_CACHE_DIR);

namespace ater {

/*
Abstract base class for build cache implementations
*/
class BuildCache {
public:
    BuildCache() = default;

    virtual ~BuildCache() = default;
    /*
    Retrieves the build cache artifacts for the given build directory,
        so that ideally no compilation needs to take place.

        Args:
            cmds (_type_): Build commands, these will be part of the hash used to calculate a lookup key
            build_dir (str): Build directory. The source files, Makefile and some other files will be hashed and
    used to determine the build cache key. from_sources_filter_func (Callable[[str], bool], optional): Filter
    function, which may be used to determine which files are being considered source files. Defaults to
    is_source.

        Returns:
            Tuple[bool, Optional[str]]: A tuple indicating whether the build cache was successfully retrieved,
    and a cache key (which should be passed on to store_build_cache on rebuild )
    */
    virtual std::tuple<bool, std::string>
    RetrieveBuildCache(const std::vector<std::string>&          cmds,
                       const std::filesystem::path&             build_dir,
                       const std::function<bool(std::string&)>& from_sources_filter_func = IsSource) = 0;

    /*
    Store the build cache artifacts

        Args:
            cmds ( List[str]): Build commands, these will be part of the hash used to calculate a lookup key
            build_dir (str): Path to build directory to retrieve build artifacts from
            cache_key (str): Cache key, as returned from retrieve_build_cache
            filter_func (Callable[[str], bool], optional): Filter function, which may be used to determine which
    files are being considered cacheable artifact files. Defaults to is_cache_artifact.

        Returns:
            bool: Whether the artifacts were successfully stored
    */
    virtual bool StoreBuildCache(const std::vector<std::string>&          cmds,
                                 const std::filesystem::path&             build_dir,
                                 const std::string&                       cache_key,
                                 const std::function<bool(std::string&)>& filter_func = IsCacheArtifact) = 0;

    // /*
    // Maybe clean up the build cache if its been longer than `cleanup_max_age_seconds` that it has been cleaned up

    //     Args:
    //         lru_retention_hours (int, optional): How many hours should unused elements be retained in the cache?
    // Defaults to 72.
    //         cleanup_max_age_seconds (int, optional): Cleanup interval in seconds. Defaults to 3600.
    // */
    // virtual void MaybeCleanup(const int lru_retention_hours = 72, const int cleanup_max_age_seconds = 3600);

    // /*
    // Do a cache cleanup.

    //     Args:
    //         retention_hours (int, optional): How many hours should unused elements be retained in the cache?
    // Defaults to 72.
    // */
    // virtual void Cleanup(const int lru_retention_hours = 72);
};

class NoBuildCache: public BuildCache {
public:
    explicit NoBuildCache();

    std::tuple<bool, std::string>
    RetrieveBuildCache(const std::vector<std::string>&          cmds,
                       const std::filesystem::path&             build_dir,
                       const std::function<bool(std::string&)>& from_sources_filter_func = IsSource) override;

    bool StoreBuildCache(const std::vector<std::string>&          cmds,
                         const std::filesystem::path&             build_dir,
                         const std::string&                       cache_key,
                         const std::function<bool(std::string&)>& filter_func = IsCacheArtifact) override;
};

class FileBasedBuildCache: public BuildCache {
public:
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
    explicit FileBasedBuildCache(const std::filesystem::path& cache_dir,
                                 const int                    lru_retention_hours     = 72,
                                 const int                    cleanup_max_age_seconds = 3600,
                                 bool                         debug                   = true);

    std::tuple<bool, std::string>
    RetrieveBuildCache(const std::vector<std::string>&          cmds,
                       const std::filesystem::path&             build_dir,
                       const std::function<bool(std::string&)>& from_sources_filter_func = IsSource);

    bool StoreBuildCache(const std::vector<std::string>&          cmds,
                         const std::filesystem::path&             build_dir,
                         const std::string&                       cache_key,
                         const std::function<bool(std::string&)>& filter_func = IsCacheArtifact);
    void MaybeCleanUp(const int lru_retention_hours = 72, const int cleanup_max_age_seconds = 3600);

    void CleanUp(const int lru_retention_hours = 72);

private:
    const std::filesystem::path cache_dir_;
    const int                   lru_retention_hours_;
    const int                   cleanup_max_age_seconds_;
    bool                        debug_;
};

inline std::shared_ptr<BuildCache> CreateBuildCache()
{
    if (FLAGS_ATER_BUILD_CACHE_DIR == "") {
        return std::make_shared<NoBuildCache>();
    }
    else {
        return std::make_shared<FileBasedBuildCache>(FLAGS_ATER_BUILD_CACHE_DIR);
    }
}

}  // namespace ater