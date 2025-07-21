#include "flashck/core/profiling/builder.h"

#include <iomanip>
#include <numeric>
#include <sstream>

#include "flashck/core/profiling/compiler.h"
#include "flashck/core/profiling/profiling_engine.h"
#include "flashck/core/utils/common.h"

FC_DECLARE_int32(FC_NUM_BUILDERS);              // Number of parallel builder jobs (-1 for auto-detect)
FC_DECLARE_bool(FC_COMBINE_PROFILING_SOURCES);  // Enable intelligent source combination
FC_DECLARE_bool(FC_FORCE_PROFILING_DB);         // Force profiling database usage
FC_DECLARE_string(FC_HOME_PATH);                // Base path for FlashCK operations

namespace flashck {

Builder::Builder()
{
    // Determine optimal number of parallel compilation jobs
    const int num_cpus = std::thread::hardware_concurrency();
    if (FLAGS_FC_NUM_BUILDERS == -1) {
        // Auto-detect: use all available CPU cores
        num_jobs_ = num_cpus;
    }
    else if (FLAGS_FC_NUM_BUILDERS < num_cpus && FLAGS_FC_NUM_BUILDERS != -1) {
        // User-specified: use the configured number if it's reasonable
        num_jobs_ = FLAGS_FC_NUM_BUILDERS;
    }
    else {
        // Fallback: use all available CPU cores
        num_jobs_ = num_cpus;
    }

    VLOG(1) << "Builder initialized with " << num_jobs_ << " parallel jobs (CPU cores: " << num_cpus << ")";
}

std::filesystem::path Builder::CombineSources(const std::set<std::filesystem::path>& sources)
{
    // Optimization: no need to combine a single source file
    if (sources.size() == 1) {
        auto single_source = *(sources.begin());
        VLOG(1) << "Single source file, no combination needed: " << single_source.string();
        return single_source;
    }

    // Combine multiple source files into a single compilation unit
    std::ostringstream combined_content;
    for (const auto& source : sources) {
        std::ifstream source_file(source);
        if (!source_file.is_open()) {
            FC_THROW(Unavailable("Unable to open source file: {}", source.string()));
        }

        // Efficiently read and append file content
        combined_content << source_file.rdbuf();
        combined_content << '\n';  // Ensure proper line separation
    }

    // Generate a unique filename based on the source file list
    std::ostringstream source_list;
    bool               first = true;
    for (const auto& source : sources) {
        if (!first) {
            source_list << ";";
        }
        source_list << source.string();
        first = false;
    }

    // Create hash-based filename to avoid collisions
    std::string combined_filename = HashToHexString(source_list.str());

    // Use round-robin directory selection for better load distribution
    static int            sources_idx = 0;
    std::filesystem::path target_dir =
        std::filesystem::path(*std::next(sources.begin(), sources_idx++ % sources.size())).parent_path();

    std::filesystem::path combined_file_path = target_dir / Sprintf("temp_{}.cc", combined_filename);
    VLOG(1) << "Combined " << sources.size() << " sources into: " << combined_file_path.string();

    // Write the combined content to the new file
    FileManager::WriteFile(combined_file_path, combined_content.str());
    return combined_file_path;
}

std::map<std::filesystem::path, std::set<std::filesystem::path>>
Builder::CombineTuningSources(const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources,
                              const int                                                               num_jobs)
{
    // Strategy 1: If we have enough targets to saturate available jobs, combine all sources per target
    if (target_to_sources.size() >= num_jobs_) {
        VLOG(1) << "Sufficient targets (" << target_to_sources.size() << ") for " << num_jobs_
                << " jobs, combining all sources per target";

        std::map<std::filesystem::path, std::set<std::filesystem::path>> target_to_combined_sources;
        for (const auto& [target, sources] : target_to_sources) {
            target_to_combined_sources[target] = std::set<std::filesystem::path>({CombineSources(sources)});
        }
        VLOG(1) << "Source combination completed for " << target_to_combined_sources.size() << " targets";
        return target_to_combined_sources;
    }

    // Strategy 2: Intelligent load balancing for optimal parallel compilation
    std::map<std::filesystem::path, std::set<std::filesystem::path>>
        multi_source_targets;    // Targets with multiple sources
    int num_multi_sources  = 0;  // Total count of source files in multi-source targets
    int num_single_sources = 0;  // Count of single-source targets

    // Categorize targets by source count
    for (const auto& [target, sources] : target_to_sources) {
        if (sources.size() > 1) {
            multi_source_targets[target] = sources;
            num_multi_sources += sources.size();
        }
        else {
            num_single_sources++;
        }
    }

    // Early exit conditions for optimization
    if (num_multi_sources == 0) {
        VLOG(1) << "All targets are single-source, no combination needed";
        return target_to_sources;
    }

    if (num_multi_sources + num_single_sources <= num_jobs_) {
        VLOG(1) << "Total source count (" << (num_multi_sources + num_single_sources) << ") <= available jobs ("
                << num_jobs_ << "), no combination needed";
        return target_to_sources;
    }

    // Calculate optimal source distribution for load balancing
    int num_combined_sources = num_jobs_ - num_single_sources;

    // Proportionally distribute combined sources among multi-source targets
    // Each target gets at least 1 combined source, with additional sources
    // allocated proportionally to their original source count
    std::map<std::filesystem::path, int> sources_per_target;
    for (const auto& [target, sources] : multi_source_targets) {
        int proportional_allocation =
            static_cast<int>(static_cast<double>(sources.size()) / num_multi_sources * num_combined_sources);
        sources_per_target[target] = std::max(proportional_allocation, 1);
    }

    // Distribute any remaining sources using round-robin allocation
    int total_allocated =
        std::accumulate(sources_per_target.begin(), sources_per_target.end(), 0, [](int sum, const auto& pair) {
            return sum + pair.second;
        });

    int remaining_sources = num_combined_sources - total_allocated;
    if (remaining_sources > 0) {
        VLOG(1) << "Distributing " << remaining_sources << " remaining sources via round-robin";

        std::vector<std::filesystem::path> target_list;
        for (const auto& [target, _] : sources_per_target) {
            target_list.emplace_back(target);
        }

        // Round-robin allocation of remaining sources
        for (int i = 0; i < remaining_sources; ++i) {
            sources_per_target[target_list[i % target_list.size()]]++;
        }
    }

    // Build the final result with intelligently combined sources
    std::map<std::filesystem::path, std::set<std::filesystem::path>> result;
    for (const auto& [target, sources] : target_to_sources) {
        if (multi_source_targets.find(target) != multi_source_targets.end()) {
            // Multi-source target: create optimized batches for parallel compilation
            int num_batches = sources_per_target[target];

            // Distribute sources across batches using round-robin for even loading
            std::vector<std::set<std::filesystem::path>> batches(num_batches);
            int                                          batch_idx = 0;

            for (const auto& source : sources) {
                batches[batch_idx].emplace(source);
                batch_idx = (batch_idx + 1) % num_batches;
            }

            // Combine sources within each batch and add to result
            for (const auto& batch : batches) {
                if (!batch.empty()) {
                    result[target].emplace(CombineSources(batch));
                }
            }

            VLOG(1) << "Target " << target.filename() << ": " << sources.size() << " sources -> " << num_batches
                    << " combined batches";
        }
        else {
            // Single-source target: use as-is for optimal compilation
            result[target] = sources;
        }
    }

    VLOG(1) << "Intelligent source combination completed: " << target_to_sources.size() << " targets optimized for "
            << num_jobs_ << " parallel jobs";
    return result;
}

std::filesystem::path
Builder::GenMakefileForRunning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                               const std::string&           so_file_name,
                               const std::filesystem::path& build_dir)
{
    // Makefile template for building shared library (.so) from object files
    // Uses pattern rules for automatic .cc -> .o compilation
    std::string makefile_tpl = "obj_files = {{obj_files}}\n\n"
                               "%.o : %.cc\n\t{{c_file_cmd}}\n\n"
                               ".PHONY: all clean clean_constants\n"
                               "all: {{targets}}\n\n"
                               "{{targets}}: $(obj_files)\n\t{{build_so_cmd}}\n\n"
                               "clean:\n\trm -f *.obj {{targets}}";

    // Helper function to extract filename from full path
    auto extract_filename = [](const std::filesystem::path& path) {
        auto path_components = SplitStrings(path.string(), "/");
        return std::filesystem::path(path_components.back());
    };

    // Extract object file names from the file tuples
    std::vector<std::string> object_files;
    object_files.reserve(file_tuples.size());

    for (const auto& [source_path, object_path] : file_tuples) {
        object_files.push_back(extract_filename(object_path).string());
    }

    // Get compiler commands using static method calls
    std::string shared_lib_cmd     = Compiler::GetCompilerCommand({"$^"}, "$@", "so");
    std::string object_compile_cmd = Compiler::GetCompilerCommand({"$<"}, "$@", "o");

    // Template substitution values
    jinja2::ValuesMap tpl_values{{"obj_files", JoinStrings(object_files, " ")},
                                 {"c_file_cmd", object_compile_cmd},
                                 {"build_so_cmd", shared_lib_cmd},
                                 {"targets", so_file_name}};

    // Generate the actual Makefile content
    std::string makefile_content = TEMPLATE_CHECK(makefile_tpl, tpl_values, "Builder::GenMakefileForRunning");
    std::string makefile_name    = Sprintf("Makefile_{}", HashToHexString(so_file_name));

    VLOG(1) << "Generated running Makefile: " << makefile_name << " (objects: " << object_files.size()
            << ", target: " << so_file_name << ")";

    // Write Makefile to build directory
    std::filesystem::path makefile_path = build_dir / makefile_name;
    FileManager::WriteFile(makefile_path, makefile_content);

    return makefile_name;
}

std::filesystem::path
Builder::GenMakefileForTuning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                              std::filesystem::path&                                                       profiler_dir)
{
    std::string makefile_tpl = "all: {{targets}}\n\n.PHONY: all clean\n\n{{commands}}\nclean:\n\trm -f {{targets}}";

    // normalize the profiler dir: add / at the end
    profiler_dir = profiler_dir / "";

    // deduplicate targets from different kernels
    std::map<std::filesystem::path, std::set<std::filesystem::path>> target_to_sources;

    for (const auto& [source, target] : file_tuples) {
        target_to_sources[target] = std::set<std::filesystem::path>({source});
    }

    // stabilize the order of sources per targets
    auto calculate_num_sources =
        [&](const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources) {
            int num_sources = 0;
            for (const auto& [_, sources] : target_to_sources) {
                num_sources += sources.size();
            }
            return num_sources;
        };

    if (FLAGS_FC_COMBINE_PROFILING_SOURCES) {
        VLOG(1) << "enable combine profiling sources";
        int  num_sources_before        = calculate_num_sources(target_to_sources);
        auto combine_target_to_sources = CombineTuningSources(target_to_sources, num_jobs_);
        int  num_sources_after         = calculate_num_sources(combine_target_to_sources);
        if (num_sources_after <= num_sources_before) {
            VLOG(1) << "Combined " << num_sources_before << " sources into " << num_sources_after << " sources";
        }
    }

    auto split_path = [](const std::filesystem::path& path) {
        auto path_parts = SplitStrings(path.string(), "/");
        if (path_parts.size() >= 3) {
            return std::filesystem::path(path_parts[path_parts.size() - 3] + "/" + path_parts[path_parts.size() - 2]
                                         + "/" + path_parts.back());
        }
        return path;
    };

    std::set<std::filesystem::path>                                  targets;
    std::map<std::filesystem::path, std::set<std::filesystem::path>> dependencies;

    for (auto& [target, sources] : target_to_sources) {
        std::filesystem::path target_file = split_path(target);
        if (sources.size() == 1) {
            // single source: no need to combine
            std::filesystem::path source = split_path(*(sources.begin()));
            dependencies[target_file]    = std::set<std::filesystem::path>({source});
            VLOG(1) << "single source: " << source.string() << " target_file: " << target_file.string();
        }
        else {
            // multi-source profiler executable
            std::set<std::filesystem::path> objects;
            for (auto& source : sources) {
                std::filesystem::path source_file = split_path(source);
                std::filesystem::path object      = source_file.replace_extension(".o");  // source ".cc"
                if (!std::filesystem::exists(profiler_dir / source_file)) {
                    // compile the object only if it is absent
                    dependencies[object] = std::set<std::filesystem::path>({source_file});
                }
                VLOG(1) << "multi-source " << source_file.string() << " profiler executable: " << target_file.string();
                objects.emplace(object);
            }
            //  then link the objects into an executable
            dependencies[target_file] = objects;
        }
        targets.emplace(target_file);
    }

    std::vector<std::string>        commands;
    int                             num_compiled_sources = 0;
    std::set<std::filesystem::path> target_names;
    for (const auto& [target, srcs] : dependencies) {
        // for each "target: srcs" pair,
        // generate two lines for the Makefile
        std::string src_str  = JoinStrings(dependencies[target], " ");
        std::string dep_line = Sprintf("{}: {}", target.string(), src_str);

        // Convert srcs to vector of strings for GetCompilerCommand
        std::vector<std::string> src_files;
        for (const auto& src : srcs) {
            src_files.push_back(src.string());
        }

        std::string cmd_line = Compiler::GetCompilerCommand(src_files, target.string(), "exe");

        std::string command = Sprintf("{}\n\t{}", dep_line, cmd_line);
        commands.emplace_back(command);

        VLOG(2) << "execute command: " << command;
        // update compilation statistics
        std::for_each(srcs.begin(), srcs.end(), [&](const std::filesystem::path& src) {
            if (EndsWith(src.string(), ".cc")) {
                num_compiled_sources += 1;
            }
        });

        if (target.extension().string() != ".o") {
            target_names.emplace(split_path(target));
        }
    }

    VLOG(1) << "compiling " << num_compiled_sources << " profiling sources";
    VLOG(1) << "linking " << target_names.size() << " profiling executables";

    jinja2::ValuesMap makefile_value_map{{"targets", JoinStrings(targets, " ")},
                                         {"commands", JoinStrings(commands, "\n")}};

    std::string makefile_content = TEMPLATE_CHECK(makefile_tpl, makefile_value_map, "Builder::GenMakefileForTuning");

    // make the Makefile name dependent on the built target names
    std::string target_names_str = JoinStrings(target_names, "_");
    std::string makefile_suffix  = HashToHexString(target_names_str);
    std::string makefile_name    = Sprintf("Makefile_{}", makefile_suffix);

    VLOG(1) << "generate makefile name for tuning: " << makefile_name;

    std::filesystem::path dumpfile = profiler_dir / makefile_name;
    FileManager::WriteFile(dumpfile, makefile_content);
    return dumpfile;
}

void Builder::MakeTuning(
    const std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>& generated_profiling_files,
    const std::string&                                                                        model_name,
    const std::string&                                                                        folder_name)
{
    bool is_empty = std::all_of(
        generated_profiling_files.begin(), generated_profiling_files.end(), [](const auto& v) { return v.empty(); });
    if (is_empty) {
        VLOG(1) << "model all kernel profiler using cache, not generate makefile to build";
        return;
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;
    for (const auto& kernel_profilers : generated_profiling_files) {
        for (const auto& [source, target] : kernel_profilers) {
            if (!std::filesystem::exists(source)) {
                FC_THROW(Unavailable("source file {} not exist", source.string()));
            }

            if (std::filesystem::exists(source) && std::filesystem::exists(target)) {
                // skip the existing target
                VLOG(1) << "source: " << source.string() << " and target: " << target.string() << " exists, skip";
                continue;
            }

            file_tuples.push_back(std::make_tuple(source, target));
        }
    }

    // generate a makefile for the profilers
    std::filesystem::path build_dir =
        std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name / "profiling";

    std::filesystem::path makefile_path = GenMakefileForTuning(file_tuples, build_dir).string();
    VLOG(1) << "Generated Makefile for profilers: " << makefile_path.string();

    std::string make_flags     = Sprintf("-f {} --output-sync -C {}", makefile_path.string(), build_dir.string());
    std::string make_clean_cmd = Sprintf("make {} clean", make_flags);
    std::string make_all_cmd   = Sprintf("make {} -j{} all", make_flags, num_jobs_);

    // Track compilation statistics for tuning
    compilation_stats_.total_compilations += file_tuples.size();
    compilation_stats_.tuning_total += file_tuples.size();

    auto [success, output] = RunMakeCmds({make_clean_cmd, make_all_cmd}, build_dir);

    if (success) {
        compilation_stats_.successful_compilations += file_tuples.size();
        compilation_stats_.tuning_successful += file_tuples.size();
        VLOG(1) << "Successfully compiled " << file_tuples.size() << " profiling sources";
    }
    else {
        auto failed_files = ParseFailedFiles(output);
        compilation_stats_.failed_compilations += failed_files.size();
        compilation_stats_.successful_compilations += (file_tuples.size() - failed_files.size());

        compilation_stats_.tuning_failed += failed_files.size();
        compilation_stats_.tuning_successful += (file_tuples.size() - failed_files.size());

        for (const auto& failed_file : failed_files) {
            compilation_stats_.failed_files.push_back(failed_file);
            compilation_stats_.tuning_failed_files.push_back(failed_file);
        }

        LOG(ERROR) << "Compilation failed for " << failed_files.size() << " out of " << file_tuples.size() << " files";
        LOG(ERROR) << "Failed files: " << JoinStrings(failed_files, ", ");
    }

    VLOG(1) << FormatCompilationStats(compilation_stats_.tuning_successful, compilation_stats_.tuning_total, "Tuning");
    VLOG(1) << FormatCompilationStats(
        compilation_stats_.successful_compilations, compilation_stats_.total_compilations, "Overall");
}

void Builder::MakeRunning(
    const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& generated_profiling_files,
    const std::string&                                                           so_file_name,
    const std::string&                                                           model_name,
    const std::string&                                                           folder_name)
{
    std::filesystem::path build_dir    = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name;
    std::filesystem::path so_file_path = build_dir / so_file_name;

    if (std::filesystem::exists(so_file_path)) {
        VLOG(1) << "model all kernel function using cache, not generate makefile to build";
        return;
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> filter_generated_files;
    for (const auto& [source, target] : generated_profiling_files) {
        if (!std::filesystem::exists(source)) {
            FC_THROW(Unavailable("source file {} not exist", source.string()));
        }

        if (std::filesystem::exists(source) && std::filesystem::exists(target)) {
            // skip the existing target
            VLOG(1) << "source: " << source.string() << " and target: " << target.string() << " exists, skip";
            continue;
        }
        filter_generated_files.push_back(std::make_tuple(source, target));
    }

    // generate a makefile for running
    std::filesystem::path makefile_path = GenMakefileForRunning(filter_generated_files, so_file_name, build_dir);
    VLOG(1) << "Generated Makefile for running: " << makefile_path.string();

    std::string make_flags     = Sprintf("-f {} --output-sync -C {}", makefile_path.string(), build_dir.string());
    std::string make_clean_cmd = Sprintf("make {} clean", make_flags);
    std::string make_all_cmd   = Sprintf("make {} -j{} all", make_flags, num_jobs_);

    // Track compilation statistics for running
    compilation_stats_.total_compilations += filter_generated_files.size();
    compilation_stats_.running_total += filter_generated_files.size();

    auto [success, output] = RunMakeCmds({make_clean_cmd, make_all_cmd}, build_dir);

    if (success) {
        compilation_stats_.successful_compilations += filter_generated_files.size();
        compilation_stats_.running_successful += filter_generated_files.size();
        VLOG(1) << "Successfully compiled " << filter_generated_files.size() << " running sources";
    }
    else {
        auto failed_files = ParseFailedFiles(output);
        compilation_stats_.failed_compilations += failed_files.size();
        compilation_stats_.successful_compilations += (filter_generated_files.size() - failed_files.size());

        compilation_stats_.running_failed += failed_files.size();
        compilation_stats_.running_successful += (filter_generated_files.size() - failed_files.size());

        for (const auto& failed_file : failed_files) {
            compilation_stats_.failed_files.push_back(failed_file);
            compilation_stats_.running_failed_files.push_back(failed_file);
        }

        LOG(ERROR) << "Compilation failed for " << failed_files.size() << " out of " << filter_generated_files.size()
                   << " files";
        LOG(ERROR) << "Failed files: " << JoinStrings(failed_files, ", ");
    }

    VLOG(1) << FormatCompilationStats(
        compilation_stats_.running_successful, compilation_stats_.running_total, "Running");
    VLOG(1) << FormatCompilationStats(
        compilation_stats_.successful_compilations, compilation_stats_.total_compilations, "Overall");
}
}  // namespace flashck
