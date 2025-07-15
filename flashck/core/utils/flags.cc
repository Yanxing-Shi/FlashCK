#include "flashck/core/utils/flags.h"

FC_DEFINE_EXPORTED_string(FC_HOME_PATH, "/test_ck_yanxishi/flash_ck", "The home path to the root of the project");

FC_DEFINE_EXPORTED_string(FC_ROCM_PATH, "/opt/rocm", "rocm path");

FC_DEFINE_EXPORTED_string(FC_TUNING_DB_DIR, "", "profiler cache dir");

FC_DEFINE_EXPORTED_string(FC_BUILD_CACHE_DIR, "", "build cache dir");

FC_DEFINE_EXPORTED_string(FC_COMPILER_OPT_LEVEL, "o3", "compiler optimization level");

FC_DEFINE_EXPORTED_bool(FC_DEBUG_KERNEL_INSTANCE, false, "debug kernel instance");

FC_DEFINE_EXPORTED_bool(FC_SAVE_TEMP_FILE, false, "save temporary files for profiling");

FC_DEFINE_EXPORTED_bool(FC_PRINT_KERNEL_SOURCE_USAGE, false, "print kernel source usage");

FC_DEFINE_EXPORTED_bool(FC_FLUSH_DENORMALS, false, "flush denormals to zero");

FC_DEFINE_EXPORTED_bool(FC_USE_FAST_MATH, true, "use fast math optimizations");

FC_DEFINE_EXPORTED_int32(FC_NUM_BUILDERS, -1, "num builders");

FC_DEFINE_EXPORTED_int32(FC_BUILDING_MAX_ATTEMPTS, 3, "max attempts for building");

FC_DEFINE_EXPORTED_int32(FC_BUILDING_TIMEOUT, 300, "timeout for building in seconds");

FC_DEFINE_EXPORTED_bool(FC_COMBINE_PROFILING_SOURCES, false, "combine multiple profiler sources per target");

FC_DEFINE_EXPORTED_bool(FC_TIME_COMPILATION, false, "time each make command at compilation time");

FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILING_DB, true, "force the profiler to use the cached results");

FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILING, true, "force profiling");

FC_DEFINE_EXPORTED_bool(FC_FLUSH_PROFILING_DB,
                        false,
                        "if flush profile cache is enabled, the profiler will flush the profile database");

FC_DEFINE_EXPORTED_int32(FC_TUNING_MODE, 1, "Mode for tuning: 0 - heuristic, 1 - autotuning, 2 - hybrid");

FC_DEFINE_EXPORTED_int32(FC_TUNING_METRIC, 0, "Profiling metrics level: 0=latency, 1=tflops, 2=bandwidth");

FC_DEFINE_EXPORTED_int32(FC_TUNING_WARM_UP, 16, "Profiling warm-up iterations");

FC_DEFINE_EXPORTED_int32(FC_TUNING_ITERATIONS, 64, "Profiling iterations");

FC_DEFINE_EXPORTED_bool(FC_TUNING_GPU_TIMER, true, "Enable GPU timer");

FC_DEFINE_EXPORTED_bool(FC_TUNING_VERIFY, false, "Disable profiling verification");

FC_DEFINE_EXPORTED_bool(FC_TUNING_LOG, false, "Disable logging");

FC_DEFINE_EXPORTED_bool(FC_TUNING_FLUSH_CACHE, true, "Flush cache for profiling");

FC_DEFINE_EXPORTED_int32(FC_TUNING_ROTATING_COUNT, 1, "Number of rotating ");

FC_DEFINE_EXPORTED_int32(FC_CALL_STACK_LEVEL, 1, "Error stack level: 0=summary, 1=+Python, 2=+C++");
