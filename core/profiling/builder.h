#pragma once

#include <iomanip>
#include <sstream>

#include "core/profiling/builder_utils.h"

namespace flashck {

/**
 * @class Builder
 * @brief Module responsible for compiling generated source code files into binary objects
 *
 * The Builder class handles the entire compilation pipeline including:
 * - Source file combination and optimization
 * - Makefile generation for tuning and running phases
 * - Parallel compilation with configurable job count
 * - Detailed compilation statistics with tuning/running distinction
 * - Error tracking and reporting
 */
class Builder {
public:
    /**
     * @brief Constructor that initializes the builder with optimal job count
     *
     * Automatically determines the number of parallel compilation jobs based on
     * system capabilities and user configuration (FC_NUM_BUILDERS flag).
     */
    explicit Builder();

    /**
     * @brief Combines multiple source files into a single source file
     * @param sources Set of source file paths to combine
     * @return Path to the combined source file
     *
     * This optimization reduces compilation overhead by combining multiple
     * small source files into larger compilation units.
     */
    std::filesystem::path CombineSources(const std::set<std::filesystem::path>& sources);

    /**
     * @brief Intelligently combines tuning sources to optimize parallel compilation
     * @param target_to_sources Mapping of target executables to their source files
     * @param num_jobs Number of parallel compilation jobs
     * @return Optimized mapping with combined sources for efficient parallel build
     *
     * This method balances the compilation load across available CPU cores by
     * combining sources strategically to maximize parallelization efficiency.
     */
    std::map<std::filesystem::path, std::set<std::filesystem::path>>
    CombineTuningSources(const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources,
                         const int                                                               num_jobs);

    /**
     * @brief Generates Makefile for tuning phase compilation
     * @param file_tuples Vector of (source, target) file path pairs
     * @param profiler_dir Directory where profiler executables will be built
     * @return Path to the generated Makefile
     *
     * Creates a Makefile optimized for building profiler executables used in
     * the tuning phase to determine optimal kernel configurations.
     */
    std::filesystem::path
    GenMakefileForTuning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                         std::filesystem::path&                                                       profiler_dir);

    /**
     * @brief Generates Makefile for running phase compilation
     * @param file_tuples Vector of (source, target) file path pairs
     * @param so_file_name Name of the shared library to be built
     * @param build_dir Directory where the shared library will be built
     * @return Path to the generated Makefile
     *
     * Creates a Makefile for building the final shared library that contains
     * the optimized kernels for production use.
     */
    std::filesystem::path
    GenMakefileForRunning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                          const std::string&                                                           so_file_name,
                          const std::filesystem::path&                                                 build_dir);

    /**
     * @brief Executes the tuning phase compilation
     * @param generated_profiling_files Vector of vectors containing (source, target) pairs for each kernel
     * @param model_name Name of the model being compiled
     * @param folder_name Base folder name for build artifacts (default: "kernel_profile")
     *
     * Compiles profiler executables used to benchmark and select optimal kernel
     * configurations. Updates tuning-specific compilation statistics.
     */
    void MakeTuning(const std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>&
                                       generated_profiling_files,
                    const std::string& model_name,
                    const std::string& folder_name = "kernel_profile");

    /**
     * @brief Executes the running phase compilation
     * @param generated_profiling_files Vector of (source, target) pairs for the final library
     * @param so_file_name Name of the shared library to build
     * @param model_name Name of the model being compiled
     * @param folder_name Base folder name for build artifacts (default: "kernel_profile")
     *
     * Compiles the final optimized shared library for production use.
     * Updates running-specific compilation statistics.
     */
    void
    MakeRunning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& generated_profiling_files,
                const std::string&                                                           so_file_name,
                const std::string&                                                           model_name,
                const std::string& folder_name = "kernel_profile");

    /**
     * @struct CompilationStats
     * @brief Comprehensive compilation statistics with tuning/running distinction
     *
     * Tracks detailed statistics for both overall compilation and separate
     * statistics for tuning and running phases to help diagnose build issues.
     */
    struct CompilationStats {
        // Overall compilation statistics
        size_t                   total_compilations      = 0;  ///< Total number of files compiled
        size_t                   successful_compilations = 0;  ///< Number of successfully compiled files
        size_t                   failed_compilations     = 0;  ///< Number of failed compilations
        std::vector<std::string> failed_files;                 ///< List of files that failed to compile

        // Tuning phase specific statistics
        size_t                   tuning_total      = 0;  ///< Total tuning files compiled
        size_t                   tuning_successful = 0;  ///< Successful tuning compilations
        size_t                   tuning_failed     = 0;  ///< Failed tuning compilations
        std::vector<std::string> tuning_failed_files;    ///< Tuning files that failed to compile

        // Running phase specific statistics
        size_t                   running_total      = 0;  ///< Total running files compiled
        size_t                   running_successful = 0;  ///< Successful running compilations
        size_t                   running_failed     = 0;  ///< Failed running compilations
        std::vector<std::string> running_failed_files;    ///< Running files that failed to compile

        /**
         * @brief Calculate overall compilation success rate
         * @return Success rate as percentage (0.0 to 100.0)
         */
        double GetSuccessRate() const
        {
            return total_compilations > 0 ? static_cast<double>(successful_compilations) / total_compilations * 100.0 :
                                            0.0;
        }

        /**
         * @brief Calculate tuning phase compilation success rate
         * @return Tuning success rate as percentage (0.0 to 100.0)
         */
        double GetTuningSuccessRate() const
        {
            return tuning_total > 0 ? static_cast<double>(tuning_successful) / tuning_total * 100.0 : 0.0;
        }

        /**
         * @brief Calculate running phase compilation success rate
         * @return Running success rate as percentage (0.0 to 100.0)
         */
        double GetRunningSuccessRate() const
        {
            return running_total > 0 ? static_cast<double>(running_successful) / running_total * 100.0 : 0.0;
        }

        /**
         * @brief Reset all compilation statistics to initial state
         */
        void Reset()
        {
            total_compilations      = 0;
            successful_compilations = 0;
            failed_compilations     = 0;
            failed_files.clear();

            tuning_total      = 0;
            tuning_successful = 0;
            tuning_failed     = 0;
            tuning_failed_files.clear();

            running_total      = 0;
            running_successful = 0;
            running_failed     = 0;
            running_failed_files.clear();
        }
    };

    /**
     * @brief Get read-only access to compilation statistics
     * @return Const reference to current compilation statistics
     */
    const CompilationStats& GetCompilationStats() const
    {
        return compilation_stats_;
    }

    /**
     * @brief Generate a detailed compilation report
     * @return Formatted string containing comprehensive compilation statistics
     *
     * Provides a human-readable summary of compilation results including
     * overall, tuning, and running phase statistics with success rates.
     */
    std::string GetCompilationReport() const
    {
        std::ostringstream report;
        report << "=== Compilation Statistics Summary ===" << std::endl;
        report << "Overall: " << compilation_stats_.successful_compilations << "/"
               << compilation_stats_.total_compilations << " success (" << std::fixed << std::setprecision(2)
               << compilation_stats_.GetSuccessRate() << "%)" << std::endl;

        if (compilation_stats_.tuning_total > 0) {
            report << "Tuning: " << compilation_stats_.tuning_successful << "/" << compilation_stats_.tuning_total
                   << " success (" << std::fixed << std::setprecision(2) << compilation_stats_.GetTuningSuccessRate()
                   << "%)" << std::endl;
        }

        if (compilation_stats_.running_total > 0) {
            report << "Running: " << compilation_stats_.running_successful << "/" << compilation_stats_.running_total
                   << " success (" << std::fixed << std::setprecision(2) << compilation_stats_.GetRunningSuccessRate()
                   << "%)" << std::endl;
        }

        if (!compilation_stats_.failed_files.empty()) {
            report << "Failed files: " << JoinStrings(compilation_stats_.failed_files, ", ") << std::endl;
        }

        return report.str();
    }

    /**
     * @brief Reset compilation statistics to initial state
     */
    void ResetCompilationStats()
    {
        compilation_stats_.Reset();
    }

private:
    int              num_jobs_;           ///< Number of parallel compilation jobs
    CompilationStats compilation_stats_;  ///< Compilation statistics tracker
};

}  // namespace flashck