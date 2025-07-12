#include "flashck/core/utils/flags.h"

FC_DEFINE_EXPORTED_string(FC_HOME_PATH,
                          "/torch6.0_rocm6.0_yanxishi/flashck",
                          "The home path to the root of the project");

FC_DEFINE_EXPORTED_string(FC_ROCM_PATH, "/opt/rocm", "rocm path");

FC_DEFINE_EXPORTED_string(FC_PROFILING_DB_DIR, "", "profiler cache dir");

FC_DEFINE_EXPORTED_string(FC_BUILD_CACHE_DIR, "", "build cache dir");

FC_DEFINE_EXPORTED_string(FC_COMPILER_OPT_LEVEL, "-O3", "compiler optimization level");

FC_DEFINE_EXPORTED_bool(FC_DEBUG_KERNEL_INSTANCE, false, "debug kernel instance");

FC_DEFINE_EXPORTED_bool(FC_SAVE_TEMP_FILE, false, "save temporary files for profiling");

FC_DEFINE_EXPORTED_bool(FC_PRINT_KERNEL_tpl_USAGE, false, "print kernel source usage");

FC_DEFINE_EXPORTED_bool(FC_FLUSH_DENORMALS, false, "flush denormals to zero");

FC_DEFINE_EXPORTED_bool(FC_USE_FAST_MATH, true, "use fast math optimizations");

FC_DEFINE_EXPORTED_int32(FC_NUM_BUILDERS, -1, "num builders");

FC_DEFINE_EXPORTED_int32(FC_BUILDING_MAX_ATTEMPTS, 3, "max attempts for building");

FC_DEFINE_EXPORTED_int32(FC_BUILDING_TIMEOUT, 300, "timeout for building in seconds");

FC_DEFINE_EXPORTED_bool(FC_COMBINE_PROFILING_tplS, false, "combine multiple profiler sources per target");

FC_DEFINE_EXPORTED_bool(FC_TIME_COMPILATION, false, "time each make command at compilation time");

FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILING_DB, false, "force the profiler to use the cached results");

FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILE, false, "force profiling");

FC_DEFINE_EXPORTED_bool(FC_FLUSH_PROFILING_DB,
                        false,
                        "if flush profile cache is enabled, the profiler will flush the profile database");

FC_DEFINE_EXPORTED_int32(FC_BUILD_CACHE_SKIP_PERCENTAGE, 30, "build cache skip percentage");

FC_DEFINE_EXPORTED_int32(FC_PROFILING_METRICS, 0, "Profiling metrics level: 0=latency, 1=tflops, 2=bandwidth");

FC_DEFINE_EXPORTED_int32(FC_PROFILING_WARM_UP, 16, "Profiling warm-up iterations");

FC_DEFINE_EXPORTED_int32(FC_PROFILING_ITERATIONS, 64, "Profiling iterations");

FC_DEFINE_EXPORTED_bool(FC_PROFILING_GPU_TIMER, true, "Enable GPU timer");

FC_DEFINE_EXPORTED_bool(FC_PROFILING_VERIFY, false, "Disable profiling verification");

FC_DEFINE_EXPORTED_bool(FC_PROFILING_LOG, false, "Disable logging");

FC_DEFINE_EXPORTED_bool(FC_PROFILING_FLUSH_CACHE, true, "Flush cache for profiling");

FC_DEFINE_EXPORTED_int32(FC_PROFILING_ROTATING_COUNT, 1, "Number of rotating ");

FC_DEFINE_EXPORTED_int32(FC_CALL_STACK_LEVEL, 1, "Error stack level: 0=summary, 1=+Python, 2=+C++");

FC_DEFINE_EXPORTED_string(FC_SELECTED_DEVICES,
                          "0",
                          "A list of device ids separated by comma, like: 0,1,2,3. "
                          "This option is useful when doing profiling. If you want to use "
                          "all visible devices, set this to an empty string."
                          "If you want to use all devices, set this to an empty string. "
                          "If you want to use a single device, set this to 0. "
                          "If you want to use multiple devices, set this to 0,1,2,3.");
