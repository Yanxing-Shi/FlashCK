#include "flashck/core/profiling/builder.h"

#include <iomanip>
#include <sstream>

#include "flashck/core/profiling/profiling_engine.h"
#include "flashck/core/utils/common.h"

FC_DECLARE_int32(FC_NUM_BUILDERS);
FC_DECLARE_bool(FC_COMBINE_PROFILING_tplS);
FC_DECLARE_bool(FC_FORCE_PROFILING_DB);
FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

Builder::Builder()
{
    const int num_cpus = std::thread::hardware_concurrency();
    if (FLAGS_FC_NUM_BUILDERS == -1) {
        num_jobs_ = num_cpus;
    }
    else if (FLAGS_FC_NUM_BUILDERS < num_cpus && FLAGS_FC_NUM_BUILDERS != -1) {
        num_jobs_ = FLAGS_FC_NUM_BUILDERS;
    }
}

std::filesystem::path Builder::CombineSources(const std::set<std::filesystem::path>& sources)
{
    if (sources.size() == 1) {
        // no need to combine a single source
        auto single_tpl = *(sources.begin());
        VLOG(1) << "no need to combine, a single source is " << single_tpl.string();
        return single_tpl;
    }

    std::ostringstream file_content;
    for (const auto& source : sources) {
        std::ifstream source_file(source);
        if (!source_file.is_open()) {
            FC_THROW(Unavailable("Unable to open source file: {}", source.string()));
        }

        // Read and append file content more efficiently
        file_content << source_file.rdbuf();
        file_content << '\n';  // Ensure proper line ending
    }

    // generate a new file name conditioned on the list of the source file names
    std::ostringstream source_stream;
    bool               first = true;
    for (const auto& source : sources) {
        if (!first)
            source_stream << ";";
        source_stream << source.string();
        first = false;
    }

    std::string file_name = HashToHexString(source_stream.str());

    static int            sources_idx = 0;
    std::filesystem::path file_dir =
        std::filesystem::path(*std::next(sources.begin(), sources_idx++ % sources.size())).parent_path();
    VLOG(1) << "Combined source file directory is " << file_dir.string();
    std::filesystem::path file_path = file_dir / Sprintf("temp_{}.cc", file_name);
    VLOG(1) << "Combined source file path is " << file_path.string();

    FileManager::WriteFile(file_path, file_content.str());
    return file_path;
}

std::map<std::filesystem::path, std::set<std::filesystem::path>>
Builder::CombineProfilingSources(const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_tpls,
                                 const int                                                               num_jobs)
{
    if (target_to_tpls.size() >= num_jobs_) {
        // there are at least as many targets as the total
        // number of sources required (or single source per
        // target is forced): combine everything
        VLOG(1) << "there are at least as many targets, combine everything";
        std::map<std::filesystem::path, std::set<std::filesystem::path>> target_to_combined_tpls;
        for (const auto& [target, sources] : target_to_tpls) {
            target_to_combined_tpls[target] = std::set<std::filesystem::path>({CombineSources(sources)});
        }
        VLOG(1) << "Combine finished";
        return target_to_combined_tpls;
    }

    std::map<std::filesystem::path, std::set<std::filesystem::path>> combine_candiates;  // multi-source targets
    int                                                              num_multi_tpls  = 0;
    int                                                              num_single_tpls = 0;
    for (const auto& [target, sources] : target_to_tpls) {
        if (sources.size() > 1) {
            combine_candiates[target] = sources;
            num_multi_tpls += sources.size();
        }
        else {
            num_single_tpls++;
        }
    }

    if (num_multi_tpls == 0) {
        // all targets are single-source: nothing to combine
        return target_to_tpls;
    }

    if (num_multi_tpls + num_single_tpls <= num_jobs) {
        // there are fewer source files than the total
        // number of sources required: no need to combine
        return target_to_tpls;
    }

    // number of sources we need for the multi-file targets
    int num_combined_tpls = num_jobs - num_single_tpls;

    // the number of combined sources per multi-source target as a
    // fraction of num_combined_tpls is proportional to the number of
    // multiple sources of the target (rounded down); ultimately, there
    // should be at least one source target (hence max(..., 1))
    std::map<std::filesystem::path, int> num_tpls_per_target;
    for (const auto& [target, sources] : combine_candiates) {
        num_tpls_per_target[target] =
            std::max(static_cast<int>(sources.size() / num_multi_tpls * num_combined_tpls), 1);
    }

    // do any sources remain after the above per-target distribution?
    int remaining_tpls =
        num_combined_tpls
        - std::accumulate(num_tpls_per_target.begin(),
                          num_tpls_per_target.end(),
                          0,
                          [](const int a, const std::pair<std::filesystem::path, int>& b) { return a + b.second; });
    if (remaining_tpls > 0) {
        // reverse-sort the targets by the remainder after rounding down:
        // prefer adding sources to the targets with a higher remainder
        // (i.e. the ones closest to getting another source)

        std::vector<std::filesystem::path> targets;
        for (const auto& [target, _] : num_tpls_per_target) {
            targets.emplace_back(target);
        }

        int target_id = 0;
        while (remaining_tpls > 0) {
            // increment the number of sources for the target
            num_tpls_per_target[targets[target_id]] += 1;
            target_id = (target_id + 1) % targets.size();
            remaining_tpls -= 1;
        }
    }

    std::map<std::filesystem::path, std::set<std::filesystem::path>> result;
    for (const auto& [target, sources] : target_to_tpls) {
        if (combine_candiates.find(target) != combine_candiates.end()) {
            // collect the sources of the target
            // in N batches by round robin
            int num_tpls = num_tpls_per_target[target];
            // TODO: form the source batches by the total number
            // of lines instead of the number of sources for more
            // even distribution of the compilation time per batch

            int                                          batch_id = 0;
            std::vector<std::set<std::filesystem::path>> batches(sources.size());

            for (auto& source : sources) {
                batches[batch_id].emplace(source);
                batch_id = (batch_id + 1) % num_tpls;
            }

            // combine the sources in each batch
            for (auto& batch : batches) {
                result[target].emplace(CombineSources(batch));
            }
        }
        else {
            // use the single-source profiler target as is
            result[target] = {target_to_tpls.at(target)};
        }
    }

    return result;
}

std::filesystem::path
Builder::GenMakefileForRunning(const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
                               const std::string&           so_file_name,
                               const std::filesystem::path& build_dir)
{
    std::string makefile_tpl =
        "CFLAGS = {{CFLAGS}}\nfPIC_flag = -fPIC\n\nobj_files = {{obj_files}}\n\n%.o : %.cc\n\t{{c_file_cmd}}\n\n.PHONY: all clean clean_constants\nall: {{targets}}\n\n{{targets}}: $(obj_files)\n\t{{build_so_cmd}}\n\nclean:\n\trm -f *.obj {{targets}}";

    std::vector<std::string> obj_files;
    std::for_each(file_tuples.begin(),
                  file_tuples.end(),
                  [&](const std::tuple<std::filesystem::path, std::filesystem::path>& val) {
                      obj_files.push_back(SplitStrings(std::get<1>(val).string(), "/").back());
                  });

    std::string build_so_cmd = ProfilingEngine::GetInstance()->GetCompiler()->GetCompilerCommand({"$@"}, {"$<"}, "so");

    std::string c_file_cmd = ProfilingEngine::GetInstance()->GetCompiler()->GetCompilerCommand({"$@"}, {"$<"}, "o");

    c_file_cmd   = TimeCmd(c_file_cmd);
    build_so_cmd = TimeCmd(build_so_cmd);

    jinja2::ValuesMap makefile_value_map{
        {"CFLAGS", JoinStrings(ProfilingEngine::GetInstance()->GetCompiler()->GetCompilerOptions())},
        {"obj_files", JoinStrings(obj_files, " ")},
        {"c_file_cmd", c_file_cmd},
        {"build_so_cmd", build_so_cmd},
        {"targets", so_file_name}};

    std::string makefile_str  = TemplateLoadAndRender(makefile_tpl, makefile_value_map);
    std::string makefile_name = Sprintf("Makefile_{}", HashToHexString(so_file_name));
    VLOG(1) << "generate makefile_name for running: " << makefile_name;

    std::filesystem::path dumpfile = build_dir / makefile_name;
    FileManager::WriteFile(dumpfile, makefile_str);

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
    std::map<std::filesystem::path, std::set<std::filesystem::path>> target_to_tpls;

    for (const auto& [source, target] : file_tuples) {
        target_to_tpls[target] = std::set<std::filesystem::path>({source});
    }

    // stabilize the order of sources per targets
    auto calculate_num_tpls =
        [&](const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_tpls) {
            int num_tpls = 0;
            for (const auto& [_, sources] : target_to_tpls) {
                num_tpls += sources.size();
            }
            return num_tpls;
        };

    if (FLAGS_FC_COMBINE_PROFILING_tplS) {
        LOG(INFO) << "enable combine profiling sources";
        int  num_tpls_before        = calculate_num_tpls(target_to_tpls);
        auto combine_target_to_tpls = CombineProfilingSources(target_to_tpls, num_jobs_);
        int  num_tpls_after         = calculate_num_tpls(combine_target_to_tpls);
        if (num_tpls_after <= num_tpls_before) {
            LOG(INFO) << "Combined " << num_tpls_before << " sources into " << num_tpls_after << " sources";
        }
    }

    auto split_path = [](const std::filesystem::path& path) {
        auto path_parts = SplitStrings(path.string(), "/");
        if (path_parts.size() >= 2) {
            return std::filesystem::path(path_parts[path_parts.size() - 2] + "/" + path_parts.back());
        }
        return path;
    };

    std::set<std::filesystem::path>                                  targets;
    std::map<std::filesystem::path, std::set<std::filesystem::path>> dependencies;

    for (auto& [target, sources] : target_to_tpls) {
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
    int                             num_compiled_tpls = 0;
    std::set<std::filesystem::path> target_names;
    for (const auto& [target, srcs] : dependencies) {
        // for each "target: srcs" pair,
        // generate two lines for the Makefile
        std::string src_str  = JoinStrings(dependencies[target], " ");
        std::string dep_line = Sprintf("{}: {}", target.string(), src_str);
        std::string cmd_line =
            TimeCmd(ProfilingEngine::GetInstance()->GetCompiler()->GetCompilerCommand({src_str}, target.string(), "o"));

        std::string command = Sprintf("{}\n\t{}", dep_line, cmd_line);
        commands.emplace_back(command);

        VLOG(2) << "execute command: " << command;
        // update compilation statistics
        std::for_each(srcs.begin(), srcs.end(), [&](const std::filesystem::path& src) {
            if (EndsWith(src.string(), ".cc")) {
                num_compiled_tpls += 1;
            }
        });

        if (target.extension().string() != ".o") {
            target_names.emplace(split_path(target));
        }
    }

    LOG(INFO) << "compiling " << num_compiled_tpls << " profiling sources";
    LOG(INFO) << "linking " << target_names.size() << " profiling executables";

    jinja2::ValuesMap makefile_value_map{{"targets", Sprintf("{}", JoinStrings(targets, " "))},
                                         {"commands", Sprintf("{}", JoinStrings(commands, "\n"))}};

    std::string makefile_content = TemplateLoadAndRender(makefile_tpl, makefile_value_map);

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
        LOG(INFO) << "model all kernel profiler using cache, not generate makefile to build";
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
                LOG(INFO) << "source: " << source.string() << " and target: " << target.string() << " exists, skip";
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

    // Track compilation statistics
    compilation_stats_.total_compilations += file_tuples.size();

    auto [success, output] = RunMakeCmds({make_clean_cmd, make_all_cmd}, build_dir);

    if (success) {
        compilation_stats_.successful_compilations += file_tuples.size();
        LOG(INFO) << "Successfully compiled " << file_tuples.size() << " profiling sources";
    }
    else {
        auto failed_files = ParseFailedFiles(output);
        compilation_stats_.failed_compilations += failed_files.size();
        compilation_stats_.successful_compilations += (file_tuples.size() - failed_files.size());

        for (const auto& failed_file : failed_files) {
            compilation_stats_.failed_files.push_back(failed_file);
        }

        LOG(ERROR) << "Compilation failed for " << failed_files.size() << " out of " << file_tuples.size() << " files";
        LOG(ERROR) << "Failed files: " << JoinStrings(failed_files, ", ");
    }

    LOG(INFO) << "Compilation statistics - Total: " << compilation_stats_.total_compilations
              << ", Success: " << compilation_stats_.successful_compilations
              << ", Failed: " << compilation_stats_.failed_compilations << ", Success rate: " << std::fixed
              << std::setprecision(2) << compilation_stats_.GetSuccessRate() << "%";
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
        LOG(INFO) << "model all kernel function using cache, not generate makefile to build";
        return;
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> filter_generated_files;
    for (const auto& [source, target] : generated_profiling_files) {
        if (!std::filesystem::exists(source)) {
            FC_THROW(Unavailable("source file {} not exist", source.string()));
        }

        if (std::filesystem::exists(source) && std::filesystem::exists(target)) {
            // skip the existing target
            LOG(INFO) << "source: " << source.string() << " and target: " << target.string() << " exists, skip";
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

    // Track compilation statistics
    compilation_stats_.total_compilations += filter_generated_files.size();

    auto [success, output] = RunMakeCmds({make_clean_cmd, make_all_cmd}, build_dir);

    if (success) {
        compilation_stats_.successful_compilations += filter_generated_files.size();
        LOG(INFO) << "Successfully compiled " << filter_generated_files.size() << " running sources";
    }
    else {
        auto failed_files = ParseFailedFiles(output);
        compilation_stats_.failed_compilations += failed_files.size();
        compilation_stats_.successful_compilations += (filter_generated_files.size() - failed_files.size());

        for (const auto& failed_file : failed_files) {
            compilation_stats_.failed_files.push_back(failed_file);
        }

        LOG(ERROR) << "Compilation failed for " << failed_files.size() << " out of " << filter_generated_files.size()
                   << " files";
        LOG(ERROR) << "Failed files: " << JoinStrings(failed_files, ", ");
    }

    LOG(INFO) << "Compilation statistics - Total: " << compilation_stats_.total_compilations
              << ", Success: " << compilation_stats_.successful_compilations
              << ", Failed: " << compilation_stats_.failed_compilations << ", Success rate: " << std::fixed
              << std::setprecision(2) << compilation_stats_.GetSuccessRate() << "%";
}
}  // namespace flashck
