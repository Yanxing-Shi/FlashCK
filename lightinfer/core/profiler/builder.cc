#include "lightinfer/core/profiler/builder.h"
#include "lightinfer/core/profiler/builder_utils.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <thread>

#include "lightinfer/core/profiler/target.h"
#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/file_utils.h"
#include "lightinfer/core/utils/flags.h"
#include "lightinfer/core/utils/jinjia2_utils.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/printf.h"
#include "lightinfer/core/utils/string_utils.h"
#include "lightinfer/core/utils/subprocess_utils.h"

LI_DECLARE_int32(LI_NUM_BUILDERS);
LI_DECLARE_bool(LI_TRACE_MAKEFILE);
LI_DECLARE_bool(LI_COMBINE_PROFILER_SOURCES);
LI_DECLARE_bool(LI_FORCE_PROFILER_CACHE);
LI_DECLARE_bool(LI_BUILD_CACHE_SKIP_PROFILER);
LI_DECLARE_string(LI_HOME_PATH);

namespace lightinfer {

Builder::Builder(int timeout): timeout_(timeout), do_trace_(FLAGS_LI_TRACE_MAKEFILE)
{
    const int num_cpus = std::thread::hardware_concurrency();
    if (FLAGS_LI_NUM_BUILDERS == -1) {
        num_jobs_ = num_cpus;
    }
    else if (FLAGS_LI_NUM_BUILDERS < num_cpus && FLAGS_LI_NUM_BUILDERS != -1) {
        num_jobs_ = FLAGS_LI_NUM_BUILDERS;
    }
}

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
        LI_THROW(Unavailable("Unable to open file:{} ", file_path.string()));
    }

    return file_path;
}

std::map<std::filesystem::path, std::set<std::filesystem::path>> Builder::CombineProfilerSources(
    const std::map<std::filesystem::path, std::set<std::filesystem::path>>& target_to_sources, const int num_jobs)
{
    if (target_to_sources.size() >= num_jobs_ || FLAGS_LI_FORCE_PROFILER_CACHE) {
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

std::filesystem::path Builder::GenMakefileForExecutors(
    const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
    const std::string&                                                           so_file_name,
    const std::filesystem::path&                                                 build_dir)
{
    std::string makefile_source =
        "CFLAGS = {{CFLAGS}}\nfPIC_flag = -fPIC\n\nobj_files = {{obj_files}}\n\n%.o : %.cc\n\t{{cfile_cmd}}\n\n.PHONY: all clean clean_constants\nall: {{targets}}\n\n{{targets}}: $(obj_files)\n\t{{build_so_cmd}}\n\nclean:\n\trm -f *.obj {{targets}}";

    std::vector<std::string> obj_files;
    std::for_each(file_tuples.begin(),
                  file_tuples.end(),
                  [&](const std::tuple<std::filesystem::path, std::filesystem::path>& val) {
                      obj_files.push_back(SplitString(std::get<1>(val).string(), "/").back());
                  });

    std::string build_so_cmd = "hipcc -shared -fPIC $(CFLAGS) -o $@ $(obj_files)";

    std::string c_file_cmd = Target::Instance()->CompileCmd("$@", "$<", false, false);

    if (do_trace_) {
        c_file_cmd   = AugmentForTrace(c_file_cmd);
        build_so_cmd = AugmentForTrace(build_so_cmd);
    }
    else {
        c_file_cmd   = TimeCmd(c_file_cmd);
        build_so_cmd = TimeCmd(build_so_cmd);
    }

    jinja2::ValuesMap makefile_value_map{{"CFLAGS", Target::Instance()->BuildComipleOptions(false)},
                                         {"obj_files", JoinToString(obj_files, " ")},
                                         {"cfile_cmd", c_file_cmd},
                                         {"build_so_cmd", build_so_cmd},
                                         {"targets", so_file_name}};

    std::string makefile_str  = TemplateLoadAndRender(makefile_source, makefile_value_map);
    std::string makefile_name = Sprintf("Makefile_{}", SHA1ToHexString(so_file_name));

    std::filesystem::path dumpfile = build_dir / makefile_name;
    std::ofstream         makefile(dumpfile.string().c_str());
    if (makefile.is_open()) {
        makefile << makefile_str;
        makefile.close();
    }
    else {
        LI_THROW(Unavailable("Unable to open file:{}", dumpfile.string()));
    }

    return makefile_name;
}

std::filesystem::path Builder::GenMakefileForProfilers(
    const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& file_tuples,
    std::filesystem::path&                                                       profiler_dir)
{
    std::string makefile_source = "all: {{targets}}\n\n.PHONY: all clean\n\n{{commands}}\nclean:\n\trm -f {{targets}}";

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

    if (FLAGS_LI_COMBINE_PROFILER_SOURCES) {
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
        auto path_res = *(result.end() - 2) + std::string("/") + result.back();
        return std::filesystem::path(path_res);
    };

    std::set<std::filesystem::path>                                  targets;
    std::map<std::filesystem::path, std::set<std::filesystem::path>> dependencies;

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

    std::vector<std::string>        commands;
    int                             num_compiled_sources = 0;
    std::set<std::filesystem::path> target_names;
    for (const auto& [target, srcs] : dependencies) {
        // for each "target: srcs" pair,
        // generate two lines for the Makefile
        std::string src_str  = JoinToString(dependencies[target], " ");
        std::string dep_line = Sprintf("{}: {}", target.string(), src_str);
        std::string cmd_line =
            Target::Instance()->CompileCmd(target.string(), src_str, target.extension().string() != ".o", true);

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
        std::for_each(srcs.begin(), srcs.end(), [&](const std::filesystem::path& src) {
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
        LI_THROW(Unavailable("Unable to open file: {}", dumpfile.string()));
    }
    return dumpfile;
}

void Builder::GenCompilerVersionFiles(const std::filesystem::path& build_dir)
{
    // exeute the shell command "hipcc --version" by using c++
    std::string version_bytes = subprocess::check_output({"hipcc", "--version"}).buf.data();

    std::filesystem::path version_file_path = build_dir / "hipcc_version.txt";

    std::ofstream version_file(version_file_path.c_str());

    if (version_file.is_open()) {
        version_file << version_bytes;
        version_file.close();
    }
    else {
        LI_THROW(Unavailable("Unable to open file:{}", version_file_path.string()));
    }
}

void Builder::MakeProfilers(
    const std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>& generated_profilers,
    const std::string&                                                                        model_name,
    const std::string&                                                                        folder_name)
{
    // generated_profilers: {{}}
    bool is_emtpy =
        std::all_of(generated_profilers.begin(), generated_profilers.end(), [](const auto& v) { return v.empty(); });
    if (is_emtpy) {
        LOG(INFO) << "model all kernel profiler using cache, not generate makefile to build";
        return;
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;
    for (const auto& kernel_profilers : generated_profilers) {
        for (const auto& [source, target] : kernel_profilers) {
            if (!std::filesystem::exists(source)) {
                LI_THROW(Unavailable("source file {} not exist", source.string()));
            }

            if (std::filesystem::exists(source) && std::filesystem::exists(target)) {
                // skip the existing target
                LOG(INFO) << "source: " << source.string() << "and target: " << target.string() << "exists, skip";
                continue;
            }

            file_tuples.push_back(std::make_tuple(source, target));
        }
    }

    // generate a makefile for the profilers
    std::filesystem::path build_dir = std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / model_name / "profiler";

    std::filesystem::path makefile_path = GenMakefileForProfilers(file_tuples, build_dir).string();
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
    RunMakeCmds({make_clean_cmd, make_all_cmd}, timeout_, build_dir, FLAGS_LI_BUILD_CACHE_SKIP_PROFILER);
}

void Builder::MakeExecutors(
    const std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>& generated_profilers,
    const std::string&                                                           so_file_name,
    const std::string&                                                           model_name,
    const std::string&                                                           folder_name)
{
    std::filesystem::path build_dir = std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / model_name;
    std::filesystem::path so_path   = build_dir / so_file_name;

    if (std::filesystem::exists(so_path)) {
        LOG(INFO) << "model all kernel function using cache, not generate makefile to build";
        return;
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> filter_generated_profilers;
    for (const auto& [source, target] : generated_profilers) {
        if (!std::filesystem::exists(source)) {
            LI_THROW(Unavailable("source file {} not exist", source.string()));
        }

        if (std::filesystem::exists(source) && std::filesystem::exists(target)) {
            // skip the existing target
            LOG(INFO) << "source: " << source.string() << "and target: " << target.string() << "exists, skip";
            continue;
        }
        filter_generated_profilers.push_back(std::make_tuple(source, target));
    }

    // generate a makefile for the executors
    std::filesystem::path makefile_path = GenMakefileForExecutors(generated_profilers, so_file_name, build_dir);
    VLOG(1) << "Generated Makefile for Executors: " << makefile_path.string() << "\n";

    // Write compiler version string(s) into build directory, so these can be used as part of cache key
    GenCompilerVersionFiles(build_dir);

    std::string make_flags     = Sprintf("-f {} --output-sync -C {}", makefile_path.string(), build_dir.string());
    std::string make_clean_cmd = Sprintf("make {} clean", make_flags);
    std::string make_all_cmd   = Sprintf("make {} -j{} all", make_flags, num_jobs_);

    RunMakeCmds({make_clean_cmd, make_all_cmd}, timeout_, build_dir, FLAGS_LI_BUILD_CACHE_SKIP_PROFILER);
}

}  // namespace lightinfer