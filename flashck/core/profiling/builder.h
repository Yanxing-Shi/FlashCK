#pragma once

#include "flashck/core/profiling/builder_utils.h"

namespace flashck {

// Builder is a module to compile generated source code files into binary objects.
class Builder {
public:
    explicit Builder();

    std::filesystem::path CombineSources(const std::set<std::filesystem::path>& sources);

    std::map<std::filesystem::path, std::set<std::filesystem::path>>
    CombineProfilingSources(const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources,
                            const int                                                               num_jobs);

    std::filesystem::path
    GenMakefileForTuning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                         std::filesystem::path&                                                       profiler_dir);

    std::filesystem::path
    GenMakefileForRunning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                          const std::string&                                                           so_file_name,
                          const std::filesystem::path&                                                 build_dir);

    void MakeTuning(const std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>&
                                       generated_profiling_files,
                    const std::string& model_name,
                    const std::string& folder_name = "kernel_profile");

    void
    MakeRunning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& generated_profiling_files,
                const std::string&                                                           so_file_name,
                const std::string&                                                           model_name,
                const std::string& folder_name = "kernel_profile");

private:
    int num_jobs_;
};

}  // namespace flashck