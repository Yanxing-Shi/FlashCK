#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace flashck {

// Builder is a module to compile generated source code files into binary objects.
class Builder {
public:
    explicit Builder(int timeout = 180);

    std::filesystem::path CombineSources(const std::set<std::filesystem::path>& sources);

    std::map<std::filesystem::path, std::set<std::filesystem::path>>
    CombineProfilerSources(const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources,
                           const int                                                               num_jobs);

    std::filesystem::path
    GenMakefileForExecutors(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                            const std::string&                                                           so_file_name,
                            const std::filesystem::path&                                                 build_dir);
    std::filesystem::path
    GenMakefileForProfilers(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                            std::filesystem::path&                                                       profiler_dir);

    // Write compiler version string(s) into build directory
    // for cache invalidation             purposes(different compiler versions
    // should not reuse same cached build artifacts)
    void GenCompilerVersionFiles(const std::filesystem::path& build_dir);

    void MakeProfilers(
        const std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>& generated_profilers,
        const std::string&                                                                        model_name,
        const std::string& folder_name = "kernel_profile");

    void MakeExecutors(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& generated_profilers,
                       const std::string&                                                           so_file_name,
                       const std::string&                                                           model_name,
                       const std::string& folder_name = "kernel_profile");

private:
    int num_jobs_;
    int timeout_;

    bool do_trace_ = false;
};

}  // namespace flashck