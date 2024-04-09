#include "ater/core/profiler/builder.h"
#include "ater/core/profiler/builder_utils.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <thread>

#include "ater/core/profiler/target.h"
#include "ater/core/utils/enforce.h"
#include "ater/core/utils/file_utils.h"
#include "ater/core/utils/flags.h"
#include "ater/core/utils/jinjia2_utils.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/printf.h"
#include "ater/core/utils/string_utils.h"
#include "ater/core/utils/subprocess_utils.h"

ATER_DECLARE_int32(ATER_NUM_BUILDERS);
ATER_DECLARE_bool(ATER_TRACE_MAKEFILE);
ATER_DECLARE_bool(ATER_COMBINE_PROFILER_SOURCES);
ATER_DECLARE_bool(ATER_FORCE_PROFILER_CACHE);
ATER_DECLARE_bool(ATER_BUILD_CACHE_SKIP_PROFILER);
ATER_DECLARE_string(ATER_HOME_PATH);

namespace ater {

Builder::Builder(int timeout): timeout_(timeout), do_trace_(FLAGS_ATER_TRACE_MAKEFILE)
{
    const int num_cpus = std::thread::hardware_concurrency();
    if (FLAGS_ATER_NUM_BUILDERS == -1) {
        num_jobs_ = num_cpus;
    }
    else if (FLAGS_ATER_NUM_BUILDERS < num_cpus && FLAGS_ATER_NUM_BUILDERS != -1) {
        num_jobs_ = FLAGS_ATER_NUM_BUILDERS;
    }
}

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
std::filesystem::path Builder::CombineSources(const std::set<std::filesystem::path>& sources)
{

    if (sources.size() == 1) {
        // no need to combine a single source
        auto single_source = *(sources.begin());
        VLOG(1) << "no need to combine, a single source is " << single_source.string();
        return single_source;
    }

    std::string file_lines;
    for (const auto& source : sources) {
        std::ifstream source_file(source.c_str());
        std::string   line;
        while (std::getline(source_file, line)) {
            // collect the original non-empty lines
            file_lines += line;
        }
        // the last line might not end with "\n"
        file_lines += "\n";
    }

    // generate a new file name conditioned on the list of the source file names
    std::string source_string = (*sources.begin()).string();
    std::for_each(std::next(sources.begin()), sources.end(), [&](const std::filesystem::path& val) {
        source_string.append(";").append(val.string());
    });

    std::string file_name = SHA1ToHexString(source_string);

    static int            sources_idx = 0;
    std::filesystem::path file_dir    = std::filesystem::path(*std::next(sources.begin(), sources_idx++)).parent_path();
    VLOG(1) << "Combined source file directory is " << file_dir.string();
    std::filesystem::path file_path = file_dir / Sprintf("temp_{}.cc", file_name);
    VLOG(1) << "Combined source file path is " << file_path.string();

    std::ofstream file(file_path.c_str());
    if (file.is_open()) {
        file << file_lines;
        file.close();
    }
    else {
        ATER_THROW(Unavailable("Unable to open file:{} ", file_path.string()));
    }

    return file_path;
}

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
std::map<std::filesystem::path, std::set<std::filesystem::path>> Builder::CombineProfilerSources(
    const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources, const int num_jobs)
{
    if (target_to_sources.size() >= num_jobs_ || FLAGS_ATER_FORCE_PROFILER_CACHE) {
        // there are at least as many targets as the total
        // number of sources required (or single source per
        // target is forced): combine everything
        VLOG(1) << "there are at least as many targets, combine everything";
        std::map<std::filesystem::path, std::set<std::filesystem::path>> target_to_combined_sources;
        for (const auto& [target, sources] : target_to_sources) {
            target_to_combined_sources[target] = std::set<std::filesystem::path>({CombineSources(sources)});
        }
        VLOG(1) << "Combine finished";
        return target_to_combined_sources;
    }

    std::map<std::filesystem::path, std::set<std::filesystem::path>> combine_candiates;  // multi-source targets
    int                                                              num_multi_sources  = 0;
    int                                                              num_single_sources = 0;
    for (const auto& [target, sources] : target_to_sources) {
        if (sources.size() > 1) {
            combine_candiates[target] = sources;
            num_multi_sources += sources.size();
        }
        else {
            num_single_sources++;
        }
    }

    if (num_multi_sources == 0) {
        // all targets are single-source: nothing to combine
        return target_to_sources;
    }

    if (num_multi_sources + num_single_sources <= num_jobs) {
        // there are fewer source files than the total
        // number of sources required: no need to combine
        return target_to_sources;
    }

    // number of sources we need for the multi-file targets
    int num_combined_sources = num_jobs - num_single_sources;

    // the number of combined sources per multi-source target as a
    // fraction of num_combined_sources is proportional to the number of
    // multiple sources of the target (rounded down); ultimately, there
    // should be at least one source target (hence max(..., 1))
    std::map<std::filesystem::path, int> num_sources_per_target;
    for (const auto& [target, sources] : combine_candiates) {
        num_sources_per_target[target] =
            std::max(static_cast<int>(sources.size() / num_multi_sources * num_combined_sources), 1);
    }

    // do any sources remain after the above per-target distribution?
    int remaining_sources =
        num_combined_sources
        - std::accumulate(num_sources_per_target.begin(),
                          num_sources_per_target.end(),
                          0,
                          [](const int a, const std::pair<std::filesystem::path, int>& b) { return a + b.second; });
    if (remaining_sources > 0) {
        // reverse-sort the targets by the remainder after rounding down:
        // prefer adding sources to the targets with a higher remainder
        // (i.e. the ones closest to getting another source)

        std::vector<std::filesystem::path> targets;
        for (const auto& [target, _] : num_sources_per_target) {
            targets.emplace_back(target);
        }

        int target_id = 0;
        while (remaining_sources > 0) {
            // increment the number of sources for the target
            num_sources_per_target[targets[target_id]] += 1;
            target_id = (target_id + 1) % targets.size();
            remaining_sources -= 1;
        }
    }

    std::map<std::filesystem::path, std::set<std::filesystem::path>> result;
    for (const auto& [target, sources] : target_to_sources) {
        if (combine_candiates.find(target) != combine_candiates.end()) {
            // collect the sources of the target
            // in N batches by round robin
            int num_sources = num_sources_per_target[target];
            // TODO: form the source batches by the total number
            // of lines instead of the number of sources for more
            // even distribution of the compilation time per batch

            int                                          batch_id = 0;
            std::vector<std::set<std::filesystem::path>> batches(sources.size());

            for (auto& source : sources) {
                batches[batch_id].emplace(source);
                batch_id = (batch_id + 1) % num_sources;
            }

            // combine the sources in each batch
            for (auto& batch : batches) {
                result[target].emplace(CombineSources(batch));
            }
        }
        else {
            // use the single-source profiler target as is
            result[target] = {target_to_sources.at(target)};
        }
    }

    return result;
}

std::filesystem::path Builder::GenMakefileForProfilers(
    const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
    std::filesystem::path&                                                       profiler_dir,
    bool                                                                         is_profile)
{
    std::string makefile_source = "all: {{targets}}\n\n.PHONY: all clean\n\n{{commands}}\nclean:\n\trm -f {{targets}}";

    // normalize the profiler dir: add / at the end
    profiler_dir = profiler_dir / "";
    // printf("profiler_dir: %s\n", profiler_dir.string().c_str());
    // deduplicate targets from different kernels
    std::map<std::filesystem::path, std::set<std::filesystem::path>> target_to_sources;

    for (const auto& [source, target] : file_tuples) {
        target_to_sources[target] = std::set<std::filesystem::path>({source});
    }

    // stabilize the order of sources per targets
    auto calculate_num_sources =
        [&](const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources) {
            int num_sources = 0;
            for (const auto& [target, sources] : target_to_sources) {
                num_sources += sources.size();
            }
            return num_sources;
        };

    if (FLAGS_ATER_COMBINE_PROFILER_SOURCES) {
        LOG(INFO) << "enable combine profiler source";
        int  num_sources_before        = calculate_num_sources(target_to_sources);
        auto combine_target_to_sources = CombineProfilerSources(target_to_sources, num_jobs_);
        int  num_sources_after         = calculate_num_sources(combine_target_to_sources);
        if (num_sources_after <= num_sources_before) {
            LOG(INFO) << "Combined " << num_sources_before << " sources into " << num_sources_after << " sources";
        }
    }

    auto split_path = [&](const std::filesystem::path& path, const std::string& delimiter) {
        std::vector<std::string> result = SplitString(path.string(), delimiter);
        // last second + end
        auto path_res = is_profile ? *(result.end() - 2) + std::string("/") + result.back() : result.back();
        return std::filesystem::path(path_res);
    };

    std::set<std::filesystem::path>                                  targets;
    std::map<std::filesystem::path, std::set<std::filesystem::path>> dependencies;
    if (is_profile) {
        for (auto& [target, sources] : target_to_sources) {
            std::filesystem::path target_file = split_path(target, "/");
            if (sources.size() == 1) {
                // single source: no need to combine
                std::filesystem::path source = split_path(*(sources.begin()), "/");
                dependencies[target_file]    = std::set<std::filesystem::path>({source});
                VLOG(1) << "single source: " << source.string() << " target_file: " << target_file.string() << "\n";
            }
            else {
                // multi-source profiler executable
                std::set<std::filesystem::path> objects;
                for (auto& source : sources) {
                    std::filesystem::path source_file = split_path(source, "/");
                    std::filesystem::path object      = source_file.replace_extension(".o");  // source ".cc"
                    if (!std::filesystem::exists(profiler_dir / source_file)) {
                        // compile the object only if it is absent
                        dependencies[object] = std::set<std::filesystem::path>({source_file});
                    }
                    VLOG(1) << "multi-source" << source_file.string() << "profiler executable:" << target_file.string()
                            << "\n ";
                    objects.emplace(object);
                }
                //  then link the objects into an executable
                dependencies[target_file] = objects;
            }
            targets.emplace(target_file);
        }
    }
    else {
        for (const auto& [target, sources] : target_to_sources) {
            std::filesystem::path target_file = split_path(target, "/").replace_extension(".o");
            std::filesystem::path source      = split_path(*(sources.begin()), "/");
            dependencies[target_file]         = std::set<std::filesystem::path>({source});
            VLOG(1) << "single source: " << source.string() << " target_file: " << target_file.string() << "\n";
            targets.emplace(target_file);
        }
    }

    std::vector<std::string>        commands;
    int                             num_compiled_sources = 0;
    std::set<std::filesystem::path> target_names;
    for (const auto& [target, srcs] : dependencies) {
        // for each "target: srcs" pair,
        // generate two lines for the Makefile
        std::string src_str  = JoinToString(dependencies[target], " ");
        std::string dep_line = Sprintf("{}: {}", target.string(), src_str);
        std::string cmd_line;
        if (is_profile) {
            cmd_line =
                Target::Instance()->CompileCmd(target.string(), src_str, target.extension().string() != ".o", true);
        }
        else {
            cmd_line =
                Target::Instance()->CompileCmd(target.string(), src_str, target.extension().string() != ".o", false);
        }
        if (do_trace_) {
            cmd_line = AugmentForTrace(cmd_line);
        }
        else {
            cmd_line = TimeCmd(cmd_line);
        }

        std::string command = Sprintf("{}\n\t{}\n", dep_line, cmd_line);
        commands.emplace_back(command);

        VLOG(2) << "execute command: " << command << "\n";
        // update compilation statistics
        for_each(srcs.begin(), srcs.end(), [&](const std::filesystem::path& src) {
            if (EndsWith(src.string(), ".cc")) {
                num_compiled_sources += 1;
            }
        });

        if (target.extension().string() != ".o") {
            target_names.emplace(split_path(target, "/"));
        }
    }

    LOG(INFO) << "compiling " << num_compiled_sources << " profiler sources";
    LOG(INFO) << "linking " << target_names.size() << " profiler executables";

    jinja2::ValuesMap makefile_value_map{{"targets", Sprintf("{}", JoinToString(targets, " "))},
                                         {"commands", Sprintf("{}", JoinToString(commands, "\n"))}};

    std::string makefile_str = TemplateLoadAndRender(makefile_source, makefile_value_map);

    // make the Makefile name dependent on the built target names
    std::string target_names_str = JoinToString(target_names, "_");
    std::string makefile_suffix  = SHA1ToHexString(target_names_str);
    std::string makefile_name    = Sprintf("Makefile_{}", makefile_suffix);

    VLOG(1) << "generate makefile_name: " << makefile_name << "\n";

    std::filesystem::path dumpfile = profiler_dir / makefile_name;
    std::ofstream         makefile(dumpfile.c_str());

    if (makefile.is_open()) {
        makefile << makefile_str;
        makefile.close();
    }
    else {
        ATER_THROW(Unavailable("Unable to open file: {}", dumpfile.string()));
    }
    return makefile_name;
}

// Write compiler version string(s) into build directory
// for cache invalidation             purposes(different compiler versions
// should not reuse same cached build artifacts)
void Builder::GenCompilerVersionFiles(const std::filesystem::path& build_dir)
{  // Write compiler version string(s)
    // into the build directory, to enable using them for cache hash determination

    // exeute the shell command "hipcc --version" by using c++
    std::string version_bytes = subprocess::check_output({"hipcc", "--version"}).buf.data();

    std::filesystem::path version_file_path = build_dir / "hipcc_version.txt";

    std::ofstream version_file(version_file_path.c_str());

    if (version_file.is_open()) {
        version_file << version_bytes;
        version_file.close();
    }
    else {
        ATER_THROW(Unavailable("Unable to open file:{}", version_file_path.string()));
    }
}

void Builder::MakeProfilers(
    const std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>& generated_profilers,
    const std::string&                                                                        model_name,
    bool                                                                                      is_profile,
    const std::string&                                                                        folder_name)
{
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;
    for (const auto& kernel_profilers : generated_profilers) {
        for (const auto& [source, target] : kernel_profilers) {
            file_tuples.push_back(std::make_tuple(source, target));
        }
    }

    // generate a makefile for the profilers
    std::filesystem::path build_dir = std::filesystem::path(FLAGS_ATER_HOME_PATH) / folder_name / model_name;
    if (is_profile)
        build_dir = build_dir / "profiler";

    std::string           makefile_name = GenMakefileForProfilers(file_tuples, build_dir, is_profile).string();
    std::filesystem::path makefile_path = build_dir / makefile_name;
    VLOG(1) << "Generated Makefile for profilers: " << makefile_path.string() << "\n";

    // Write compiler version string(s) into build directory, so these can be used as part of cache key
    GenCompilerVersionFiles(build_dir);

    // hash all .bin files and write hash into it, so we can use their hash to build the cache key,
    // even if we delete the actual .bin file afterwards
    WriteBinHash(build_dir);

    std::string make_flags     = Sprintf("-f {} --output-sync -C {}", makefile_path.string(), build_dir.string());
    std::string make_clean_cmd = Sprintf("make {} clean", make_flags);
    // std::vector<std::string> make_clean_cmd_vec = ToVector(make_clean_cmd, " ");

    std::string make_all_cmd = Sprintf("make {} -j{} all", make_flags, num_jobs_);

    // std::vector<std::string> make_all_cmd_vec = ToVector(make_all_cmd, " ");
    RunMakeCmds({make_clean_cmd, make_all_cmd}, timeout_, build_dir, FLAGS_ATER_BUILD_CACHE_SKIP_PROFILER);
}

}  // namespace ater