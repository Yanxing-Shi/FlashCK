#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ater {

/*
Builder is a module to compile generated source code
files into binary objects.
*/
class Builder {
public:
    /*
    "Initialize a parallel builder for compiling source code.
    n_jobs: int,
        optional Run how many parallel compiling job, by default - 1,
        which will set n_jobs to `multiprocessing.cpu_count()`
    timeout: int,
        optional Timeout value,
        by default 180(seconds) */
    explicit Builder(int timeout = 180);

    /*
    Combine multiple source files (given by path) into one
        source file and return the path of the combined file.

        Parameters
        ----------
        sources : Iterable[str]
            The list of paths to the source files to combine.

        Returns
        -------
        path : str
            The path to the combined source file.
    */
    std::filesystem::path CombineSources(const std::set<std::filesystem::path>& sources);

    /*
    Combine multiple profiler sources generated for different targets
        to optimize the overall compilation time, given the available number
        of builders (CPUs). The total number of sources (across all targets)
        is set equal to the `num_builders`. Single-source targets are kept
        as is; multi-source targetss' sources are possibly combined.

        Simplifying assumptions:

            - Individual split (multiple) sources per target take
              approximately equal time to compile across different
              targets (this is, in particular, not true for the main
              profiler source file vs kernel-specific source files:
              the former is typically larger than the latter);
            - Compilation time grows linearly in the number of
              separate sources combined into a single file.

        Parameters
        ----------
        target_to_soruces : dict[str, Iterable[str]]
            The mapping from each target name to the list of sources
            required to compile this target. There can be one or more
            sources for each target.
        num_builders : int
            The number of available builders (CPUs).

        Returns
        ----------
        target_to_combined_sources : dict[str, Iterable[str]]
            Like `target_to_sources`, but with some of the source paths
            in the values replaced by the paths to the respective combined
            source files. Whether and which of the sources are combined
            depends on the arguments.
        """
    */
    std::map<std::filesystem::path, std::set<std::filesystem::path>>
    CombineProfilerSources(const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources,
                           const int                                                               num_jobs);

    std::filesystem::path
    GenMakefileForProfilers(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                            std::filesystem::path&                                                       profiler_dir,
                            bool                                                                         is_profile);

    // Write compiler version string(s) into build directory
    // for cache invalidation             purposes(different compiler versions
    // should not reuse same cached build artifacts)
    void GenCompilerVersionFiles(const std::filesystem::path& build_dir);

    void MakeProfilers(
        const std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>& generated_profilers,
        const std::string&                                                                        model_name,
        bool                                                                                      is_profile = true,
        const std::string& folder_name = "kernel_profile");

private:
    int num_jobs_;
    int timeout_;

    bool do_trace_ = false;
};

}  // namespace ater