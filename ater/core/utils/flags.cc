#include "ater/core/utils/flags.h"

#include <mutex>
#include <string>

namespace ater {

const ExportedFlagInfoMap& GetExportedFlagInfoMap()
{
    return *GetMutableExportedFlagInfoMap();
}

ExportedFlagInfoMap* GetMutableExportedFlagInfoMap()
{
    static ExportedFlagInfoMap g_exported_flag_info_map;
    return &g_exported_flag_info_map;
}

bool SetFlagValue(const char* name, const char* value)
{
    std::string ret = gflags::SetCommandLineOption(name, value);
    return ret.empty() ? false : true;
}

bool FindFlag(const char* name)
{
    std::string value;
    return gflags::GetCommandLineOption(name, &value);
}

std::once_flag g_gflags_init_flag;

void InitGflags(int argc, char** argv, bool remove_flags)
{
    std::call_once(g_gflags_init_flag, [&]() {
        // gflags::AllowCommandLineReparsing();
        gflags::ParseCommandLineFlags(&argc, &argv, true);
    });
}

}  // namespace ater

/********************************define and export flags *****************************************************/

/*------------------------------------path---------------------------*/
// The path to the root of the project
ATER_DEFINE_EXPORTED_string(ATER_HOME_PATH, "/ater_test_yanxishi/ATER/", "The home path to the root of the project");

// rocm path
ATER_DEFINE_EXPORTED_string(ATER_ROCM_PATH, "/opt/rocm", "rocm path");

// profiler cache dir
ATER_DEFINE_EXPORTED_string(ATER_PROFILER_CACHE_DIR, "", "profiler cache dir");

// build cache dir
ATER_DEFINE_EXPORTED_string(ATER_BUILD_CACHE_DIR, "", "build cache dir");

/*-----------------------------------profiler----------------------------*/
// #The reason : it is typical in our situation that an option
// #-- optimize<level>(- Ox) is for a HOST compiler.And - O3 does
// #literally nothing except for the enormous compilation time.
// #
// #So, it is safe to allow users to override this option in order
// #to significantly speedup the computations / testing, especially
// #for very large               models.
ATER_DEFINE_EXPORTED_string(ATER_COMPILER_OPT_LEVEL, "-O3", "rocm path");

// num builders
ATER_DEFINE_EXPORTED_int32(ATER_NUM_BUILDERS, -1, "num builders");

// whether if Trace makefile execution
ATER_DEFINE_EXPORTED_bool(ATER_TRACE_MAKEFILE, false, "trace makefile execution");

// Whether to combine multiple profiler sources per target
ATER_DEFINE_EXPORTED_bool(ATER_COMBINE_PROFILER_SOURCES, false, "combine multiple profiler sources per target");

// time each make command at compilation time. This helps us doing compilation time analysis.Requires to install "time".
ATER_DEFINE_EXPORTED_bool(ATER_TIME_COMPILATION, false, "time each make command at compilation time");

// Force the profiler to use the cached results. The profiler will throw a runtime exception if it cannot find cached
// results. This env may be useful to capture any cache misses due to cache version updates or other relevant code
// changes.
ATER_DEFINE_EXPORTED_bool(ATER_FORCE_PROFILER_CACHE, false, "force the profiler to use the cached results");

// boolean value of ATER_BUILD_CACHE_SKIP_PROFILER environment variable.
// Will return True if that variable is not set, if it is equal to "0",
// an empty string or "False"(case insensitive).Will return True
// in all other cases.
ATER_DEFINE_EXPORTED_bool(ATER_BUILD_CACHE_SKIP_PROFILER, false, "build cache skip profiler");

// Whether to force profile. Force profiling regardless in_ci_env, disable_profiler_codegen
ATER_DEFINE_EXPORTED_bool(ATER_FORCE_PROFILE, true, "force profiling");

// If flush profile is enabled, the profiler will flush the profile database.
ATER_DEFINE_EXPORTED_bool(ATER_FLUSH_PROFILE_CACHE,
                          false,
                          "if flush profile cache is enabled, the profiler will flush the profile database");

// When set to a non-empty string, and if ATER_BUILD_CACHE_DIR
// is set, the build cache will be skipped randomly with a probability correspinding to the specified percentage
ATER_DEFINE_EXPORTED_int32(ATER_BUILD_CACHE_SKIP_PERCENTAGE, 30, "build cache skip percentage");

/**
 * Debug related FLAG
 * Note: Used to debug. Determine the call stack to print when error or
 * exeception happens.
 * If FLAGS_call_stack_level == 0, only the error message summary will be shown.
 * If FLAGS_call_stack_level == 1, the python stack and  error message summary
 * will be shown.
 * If FLAGS_call_stack_level == 2, the python stack, c++ stack, and error
 * message summary will be shown.
 */

ATER_DEFINE_EXPORTED_int32(call_stack_level,
                           1,
                           "Determine the call stack to print when error or exeception happens."
                           "If FLAGS_call_stack_level == 0, only the error message summary will be "
                           "If FLAGS_call_stack_level == 1, the python stack and error message "
                           "summary will be shown."
                           "If FLAGS_call_stack_level == 2, the python stack, c++ stack, and "
                           "error message summary will be shown.");

/**
 * CUDA related FLAG
 * Name: FLAGS_selected_gpus
 * Value Range: integer list separated by comma, default empty list
 * Example: FLAGS_selected_gpus=0,1,2,3,4,5,6,7 to train or predict with 0~7 gpu
 * cards
 * Note: A list of device ids separated by comma, like: 0,1,2,3
 */
ATER_DEFINE_EXPORTED_string(selected_gpus,
                            "0",
                            "A list of device ids separated by comma, like: 0,1,2,3. "
                            "This option is useful when doing profiling. If you want to use "
                            "all visible devices, set this to empty string");

/**
 * Distributed related FLAG
 * Name: FLAGS_dist_threadpool_size
 * Value Range: int32, default=0
 * Example:
 * Note: Control the number of threads used for distributed modules.
 *       If it is not set, it is set to a hard thread.
 */
ATER_DEFINE_EXPORTED_int32(dist_threadpool_size, 1, "number of threads used for distributed executed.");
