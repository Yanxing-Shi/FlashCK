#include "lightinfer/core/utils/flags.h"

#include <mutex>
#include <string>

namespace lightinfer {

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

}  // namespace lightinfer

/********************************define and export flags *****************************************************/

/*------------------------------------path---------------------------*/
// The path to the root of the project
LI_DEFINE_EXPORTED_string(LI_HOME_PATH,
                          "/torch6.0_rocm6.0_yanxishi/LightInfer",
                          "The home path to the root of the project");

// rocm path
LI_DEFINE_EXPORTED_string(LI_ROCM_PATH, "/opt/rocm", "rocm path");

// profiler cache dir
LI_DEFINE_EXPORTED_string(LI_PROFILER_CACHE_DIR, "", "profiler cache dir");

// build cache dir
LI_DEFINE_EXPORTED_string(LI_BUILD_CACHE_DIR, "", "build cache dir");

/*-----------------------------------profiler----------------------------*/
// #The reason : it is typical in our situation that an option
// #-- optimize<level>(- Ox) is for a HOST compiler.And - O3 does
// #literally nothing except for the enormous compilation time.
// #
// #So, it is safe to allow users to override this option in order
// #to significantly speedup the computations / testing, especially
// #for very large               models.

LI_DEFINE_EXPORTED_string(LI_COMPILER_OPT_LEVEL, "-O3", "compiler opt level");

// LI_DEFINE_EXPORTED_bool(LI_PRINT_KERNEL_RESOURCE_USAGE, false, "print kernel resource usage");

// LI_DEFINE_EXPORTED_bool(LI_FLUSH_DENORMALS, false, "flush denormals");

// num builders
LI_DEFINE_EXPORTED_int32(LI_NUM_BUILDERS, -1, "num builders");

// whether if Trace makefile execution
LI_DEFINE_EXPORTED_bool(LI_TRACE_MAKEFILE, false, "trace makefile execution");

// Whether to combine multiple profiler sources per target
LI_DEFINE_EXPORTED_bool(LI_COMBINE_PROFILER_SOURCES, false, "combine multiple profiler sources per target");

// time each make command at compilation time. This helps us doing compilation time analysis.Requires to install "time".
LI_DEFINE_EXPORTED_bool(LI_TIME_COMPILATION, false, "time each make command at compilation time");

// Force the profiler to use the cached results. The profiler will throw a runtime exception if it cannot find cached
// results. This env may be useful to capture any cache misses due to cache version updates or other relevant code
// changes.
LI_DEFINE_EXPORTED_bool(LI_FORCE_PROFILER_CACHE, false, "force the profiler to use the cached results");

// boolean value of LI_BUILD_CACHE_SKIP_PROFILER environment variable.
// Will return True if that variable is not set, if it is equal to "0",
// an empty string or "False"(case insensitive).Will return True
// in all other cases.
LI_DEFINE_EXPORTED_bool(LI_BUILD_CACHE_SKIP_PROFILER, false, "build cache skip profiler");

// Whether to force profile. Force profiling regardless in_ci_env, disable_profiler_codegen
LI_DEFINE_EXPORTED_bool(LI_FORCE_PROFILE, false, "force profiling");

// If flush profile is enabled, the profiler will flush the profile database.
LI_DEFINE_EXPORTED_bool(LI_FLUSH_PROFILE_CACHE,
                        false,
                        "if flush profile cache is enabled, the profiler will flush the profile database");

// When set to a non-empty string, and if LI_BUILD_CACHE_DIR
// is set, the build cache will be skipped randomly with a probability correspinding to the specified percentage
LI_DEFINE_EXPORTED_int32(LI_BUILD_CACHE_SKIP_PERCENTAGE, 30, "build cache skip percentage");

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

LI_DEFINE_EXPORTED_int32(call_stack_level,
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
LI_DEFINE_EXPORTED_string(selected_gpus,
                          "0",
                          "A list of device ids separated by comma, like: 0,1,2,3. "
                          "This option is useful when doing profiling. If you want to use "
                          "all visible devices, set this to empty string");
