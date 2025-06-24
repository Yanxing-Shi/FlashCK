#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <tuple>

#include "flashck/core/profiler/build_cache_utils.h"
#include "flashck/core/utils/flags.h"

LI_DECLARE_string(LI_BUILD_CACHE_DIR);

namespace flashck {

class BuildCache {
public:
    BuildCache() = default;

    virtual ~BuildCache() = default;

    virtual std::tuple<bool, std::string>
    RetrieveBuildCache(const std::vector<std::string>&          cmds,
                       const std::filesystem::path&             build_dir,
                       const std::function<bool(std::string&)>& from_sources_filter_func = IsSource) = 0;

    virtual bool StoreBuildCache(const std::vector<std::string>&          cmds,
                                 const std::filesystem::path&             build_dir,
                                 const std::string&                       cache_key,
                                 const std::function<bool(std::string&)>& filter_func = IsCacheArtifact) = 0;
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
    explicit FileBasedBuildCache(const std::filesystem::path& cache_dir,
                                 const int                    lru_retention_hours     = 72,
                                 const int                    cleanup_max_age_seconds = 3600,
                                 bool                         debug                   = true);

    std::tuple<bool, std::string>
    RetrieveBuildCache(const std::vector<std::string>&          cmds,
                       const std::filesystem::path&             build_dir,
                       const std::function<bool(std::string&)>& from_sources_filter_func = IsSource) override;

    bool StoreBuildCache(const std::vector<std::string>&          cmds,
                         const std::filesystem::path&             build_dir,
                         const std::string&                       cache_key,
                         const std::function<bool(std::string&)>& filter_func = IsCacheArtifact) override;
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
    if (FLAGS_LI_BUILD_CACHE_DIR == "") {
        return std::make_shared<NoBuildCache>();
    }
    else {
        return std::make_shared<FileBasedBuildCache>(FLAGS_LI_BUILD_CACHE_DIR);
    }
}

}  // namespace flashck