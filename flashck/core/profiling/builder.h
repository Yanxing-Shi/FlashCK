#pragma once

#include "flashck/core/profiling/builder_utils.h"

namespace flashck {

// Builder is a module to compile generated source code files into binary objects.
class Builder {
public:
    explicit Builder();

    std::filesystem::path CombineSources(const std::set<std::filesystem::path>& sources);

    std::map<std::filesystem::path, std::set<std::filesystem::path>>
    CombineProfilingSources(const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_tpls,
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

    // Compilation statistics
    struct CompilationStats {
        size_t                   total_compilations      = 0;
        size_t                   successful_compilations = 0;
        size_t                   failed_compilations     = 0;
        std::vector<std::string> failed_files;

        double GetSuccessRate() const
        {
            return total_compilations > 0 ? static_cast<double>(successful_compilations) / total_compilations * 100.0 :
                                            0.0;
        }

        void Reset()
        {
            total_compilations      = 0;
            successful_compilations = 0;
            failed_compilations     = 0;
            failed_files.clear();
        }
    };

    // Get compilation statistics
    const CompilationStats& GetCompilationStats() const
    {
        return compilation_stats_;
    }

    // Reset compilation statistics
    void ResetCompilationStats()
    {
        compilation_stats_.Reset();
    }

private:
    int              num_jobs_;
    CompilationStats compilation_stats_;  // Track compilation statistics
};

}  // namespace flashck