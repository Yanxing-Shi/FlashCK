#include "flashck/core/utils/flags.h"

/**
 * @var FC_HOME_PATH
 * @brief Root directory of the flashck project installation
 * @details Must contain framework binaries and resource files.
 *          Default assumes standard deployment layout.
 * @warning Changing this at runtime may cause subsystem reinitialization
 * @note Environment variable: flashck_HOME overrides this value
 */
FC_DEFINE_EXPORTED_string(FC_HOME_PATH,
                          "/torch6.0_rocm6.0_yanxishi/flashck",
                          "The home path to the root of the project");

/**
 * @var FC_ROCM_PATH
 * @brief ROCm SDK installation directory
 * @details Used for GPU acceleration and kernel compilation.
 *          Must match the ROCm version used for framework compilation.
 * @note Autodetected in this priority order:
 *       1. Explicitly set value
 *       2. ROCM_PATH environment variable
 *       3. Default /opt/rocm
 */
FC_DEFINE_EXPORTED_string(FC_ROCM_PATH, "/opt/rocm", "rocm path");

/**
 * @var FC_PROFILER_CACHE_DIR
 * @brief Storage location for performance profiling data
 * @details When empty:
 *          - Defaults to $FC_HOME_PATH/profiler_cache during runtime
 *          - Automatically created if missing
 * @warning Avoid using network-mounted paths for performance critical operations
 */
FC_DEFINE_EXPORTED_string(FC_PROFILER_CACHE_DIR, "", "profiler cache dir");

/**
 * @var FC_BUILD_CACHE_DIR
 * @brief Cache directory for compiled kernels and intermediate files
 * @details When empty:
 *          - Defaults to $FC_HOME_PATH/build_cache
 *          - Must be writable by application process
 * @recommendation Use fast local storage for best performance
 */
FC_DEFINE_EXPORTED_string(FC_BUILD_CACHE_DIR, "", "build cache dir");

/**
 * @var FC_COMPILER_OPT_LEVEL
 * @brief Compiler optimization level for kernel generation
 * @details Default: -O3 (aggressive optimization). Override this to:
 *          - Reduce compilation time during development/testing
 *          - Bypass host compiler's ineffective optimization
 *          - Debug optimized code issues
 *
 * @warning Higher levels (-O3) may:
 *          - Increase compilation time exponentially
 *          - Provide negligible runtime gains for GPU kernels
 *          - Introduce non-deterministic behavior
 *
 * @note Environment override example:
 *       `export FC_COMPILER_OPT_LEVEL="-O1"`
 *
 * @recommendation For large model development:
 *               - Use "-O0" or "-O1" for iterative testing
 *               - Reserve "-O3" for final production builds
 *
 * @see FC_BUILD_CACHE_DIR for cache management of optimized binaries
 */
FC_DEFINE_EXPORTED_string(FC_COMPILER_OPT_LEVEL, "-O3", "compiler opt level");

// FC_DEFINE_EXPORTED_bool(FC_PRINT_KERNEL_RESOURCE_USAGE, false, "print kernel resource usage");

// FC_DEFINE_EXPORTED_bool(FC_FLUSH_DENORMALS, false, "flush denormals");

/**
 * @var FC_NUM_BUILDERS
 * @brief Maximum parallel compilation processes
 * @details Default (-1) auto-detects based on available CPU cores.
 *          Set to:
 *          - 1 for serial execution (debugging)
 *          - N for explicit parallelism
 * @warning Exceeding physical core count may cause thrashing
 * @note Memory constrained systems should use lower values
 * @see FC_TIME_COMPILATION for performance analysis
 */
FC_DEFINE_EXPORTED_int32(FC_NUM_BUILDERS, -1, "num builders");

/**
 * @var FC_TRACE_MAKEFILE
 * @brief Enable verbose tracing of Makefile execution
 * @details When true:
 *          - Logs all Makefile variable expansions
 *          - Records implicit rule searches
 *          - Captures recipe execution order
 * @warning Significantly impacts build performance
 * @recommendation Use with FC_NUM_BUILDERS=1 for ordered logs
 */
FC_DEFINE_EXPORTED_bool(FC_TRACE_MAKEFILE, false, "trace makefile execution");

/**
 * @var FC_COMBINE_PROFILER_SOURCES
 * @brief Aggregate profiling data from multiple sources
 * @details When enabled:
 *          - Merges profile outputs per target
 *          - Reduces file I/O overhead
 *          - Loses per-source granularity
 * @recommendation Enable for production profiling
 * @note Disable when debugging individual kernel performance
 */
FC_DEFINE_EXPORTED_bool(FC_COMBINE_PROFILER_SOURCES, false, "combine multiple profiler sources per target");

/**
 * @var FC_TIME_COMPILATION
 * @brief Measure precise compilation phase durations
 * @details Requires GNU 'time' 1.7+ in PATH:
 *          - Measures user/system/real times
 *          - Tracks memory usage peaks
 *          - Outputs to FC_PROFILER_CACHE_DIR/time.log
 * @warning Adds 5-15% compilation overhead
 * @note Combine with FC_TRACE_MAKEFILE for full critical path analysis
 */
FC_DEFINE_EXPORTED_bool(FC_TIME_COMPILATION, false, "time each make command at compilation time");

/**
 * @var FC_FORCE_PROFILER_CACHE
 * @brief Enforce usage of cached profiler data
 * @details When enabled:
 *          - Requires existence of valid profiler cache entries
 *          - Throws std::runtime_error if cached data is unavailable
 *          - Bypasses normal profiler execution path
 * @warning Using outdated cache may produce misleading results
 * @note Useful for:
 *        - Validating cache compatibility after framework updates
 *        - Reproducing historical performance issues
 */
FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILER_CACHE, false, "force the profiler to use the cached results");

/**
 * @var FC_BUILD_CACHE_SKIP_PROFILER
 * @brief Conditional profiler bypass for cached builds
 * @details Environment variable interpretation:
 *          - "0", "False" (case-insensitive), or empty → false (run profiler)
 *          - Any other non-empty value → true (skip profiler)
 * @note Typical use cases:
 *       - Accelerating CI/CD pipeline executions
 *       - Reducing noise during non-performance debugging
 * @warning Disabling profiling may mask optimization opportunities
 */
FC_DEFINE_EXPORTED_bool(FC_BUILD_CACHE_SKIP_PROFILER, false, "build cache skip profiler");

/**
 * @var FC_FORCE_PROFILE
 * @brief Unconditional profiling override
 * @details When activated:
 *          - Ignores CI environment detection
 *          - Overrides profiler-disabling flags
 *          - Forces instrumentation in all build modes
 * @warning May significantly impact:
 *          - Build performance (15-30% overhead)
 *          - Binary size (instrumentation bloat)
 * @recommendation Use with FC_FORCE_PROFILER_CACHE=false for fresh data
 */
FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILE, false, "force profiling");

/**
 * @var FC_FLUSH_PROFILE_CACHE
 * @brief Controls profile database persistence
 * @details When enabled:
 *          - Forces immediate write of profile data to persistent storage
 *          - Clears in-memory profile buffers after flush
 *          - Increases I/O operations during profiling shutdown
 * @warning Enabling may:
 *          - Add 5-15% overhead to profiling teardown
 *          - Cause data loss if interrupted mid-flush
 * @recommendation Use for:
 *               - Critical profile data preservation
 *               - Debugging profiling instrumentation errors
 * @note Combines with FC_FORCE_PROFILER_CACHE to force cache reload
 */
FC_DEFINE_EXPORTED_bool(FC_FLUSH_PROFILE_CACHE,
                        false,
                        "if flush profile cache is enabled, the profiler will flush the profile database");

/**
 * @var FC_BUILD_CACHE_SKIP_PERCENTAGE
 * @brief Probabilistic build cache bypass
 * @details Operational rules:
 *          - Requires FC_BUILD_CACHE_DIR to be configured
 *          - Percentage interpreted as integer (0-100)
 *          - Applied per build operation using Monte Carlo sampling
 * @warning Dangerous settings:
 *          - Values >75 may trigger cache thrashing
 *          - Values=100 effectively disables cache
 * @recommendation Use for:
 *               - Testing cache fallback mechanisms
 *               - Simulating cold cache scenarios
 * @note Actual skip rate follows ±2% tolerance due to PRNG characteristics
 */
FC_DEFINE_EXPORTED_int32(FC_BUILD_CACHE_SKIP_PERCENTAGE, 30, "build cache skip percentage");

/**
 * @brief Controls error stack trace verbosity
 *
 * - 0: Error summary only
 * - 1: + Python stack (default)
 * - 2: + C++ stack
 *
 * @code
 * FLAGS_call_stack_level = 2;  // Full debug
 * FLAGS_call_stack_level = 0;  // Production
 * @endcode
 *
 * @warning Level 2 impacts performance
 */
FC_DEFINE_EXPORTED_int32(call_stack_level, 1, "Error stack level: 0=summary, 1=+Python, 2=+C++");

/**
 * @var selected_gpus
 * @ingroup device_config
 * @brief Specifies visible HIP devices for computation
 * @details Core characteristics:
 *          - Accepts comma-separated integer device IDs (e.g., "0,1,2")
 *          - Default value "0" selects first available device
 *          - Empty string enables all detectable HIP devices
 * @recommendation Usage patterns:
 *               - Multi-GPU : "0,1,2,3"
 *               - Single device: "0"
 */
FC_DEFINE_EXPORTED_string(selected_gpus,
                          "0",
                          "A list of device ids separated by comma, like: 0,1,2,3. "
                          "This option is useful when doing profiling. If you want to use "
                          "all visible devices, set this to empty string");
