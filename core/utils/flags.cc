#include "core/utils/flags.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace flashck {

bool SetFlagValue(const char* name, const char* value)
{
    return !gflags::SetCommandLineOption(name, value).empty();
}

bool FindFlag(const char* name)
{
    std::string dummy;
    return gflags::GetCommandLineOption(name, &dummy);
}

bool GetFlagValue(const char* name, std::string& value)
{
    return gflags::GetCommandLineOption(name, &value);
}

void PrintAllFlags(bool writable_only)
{
    const auto& flag_map = GetExportedFlagInfoMap();

    std::cout << "\n=== FlashCK Flags ===\n";
    std::cout << std::left << std::setw(30) << "Flag Name" << std::setw(15) << "Current Value" << std::setw(15)
              << "Default Value" << std::setw(8) << "Writable"
              << "Description\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& [name, info] : flag_map) {
        if (writable_only && !info.is_writable) {
            continue;
        }

        std::string current_value;
        GetFlagValue(name.c_str(), current_value);

        std::string default_str;
        std::visit(
            [&default_str](const auto& val) {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, bool>) {
                    default_str = val ? "true" : "false";
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    default_str = val;
                }
                else {
                    default_str = std::to_string(val);
                }
            },
            info.default_value);

        std::cout << std::left << std::setw(30) << name << std::setw(15) << current_value << std::setw(15)
                  << default_str << std::setw(8) << (info.is_writable ? "Yes" : "No") << info.doc << "\n";
    }
    std::cout << "\n";
}

bool PrintFlagInfo(const char* name)
{
    const auto& flag_map = GetExportedFlagInfoMap();
    auto        it       = flag_map.find(name);

    if (it == flag_map.end()) {
        std::cout << "Flag '" << name << "' not found.\n";
        return false;
    }

    const auto& info = it->second;
    std::string current_value;
    GetFlagValue(name, current_value);

    std::cout << "\n=== Flag Information ===\n";
    std::cout << "Name: " << info.name << "\n";
    std::cout << "Current Value: " << current_value << "\n";

    std::cout << "Default Value: ";
    std::visit(
        [](const auto& val) {
            if constexpr (std::is_same_v<std::decay_t<decltype(val)>, bool>) {
                std::cout << (val ? "true" : "false");
            }
            else {
                std::cout << val;
            }
        },
        info.default_value);
    std::cout << "\n";

    std::cout << "Type: ";
    std::visit(
        [](const auto& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, bool>) {
                std::cout << "bool";
            }
            else if constexpr (std::is_same_v<T, int32_t>) {
                std::cout << "int32_t";
            }
            else if constexpr (std::is_same_v<T, int64_t>) {
                std::cout << "int64_t";
            }
            else if constexpr (std::is_same_v<T, uint64_t>) {
                std::cout << "uint64_t";
            }
            else if constexpr (std::is_same_v<T, double>) {
                std::cout << "double";
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                std::cout << "string";
            }
            else {
                std::cout << "unknown";
            }
        },
        info.default_value);
    std::cout << "\n";

    std::cout << "Writable: " << (info.is_writable ? "Yes" : "No") << "\n";
    std::cout << "Description: " << info.doc << "\n\n";

    return true;
}

std::map<std::string, FlagInfo> GetAllFlagInfo(bool writable_only)
{
    const auto&                     flag_map = GetExportedFlagInfoMap();
    std::map<std::string, FlagInfo> result;

    for (const auto& [name, info] : flag_map) {
        if (writable_only && !info.is_writable) {
            continue;
        }
        result[name] = info;
    }

    return result;
}

}  // namespace flashck

// ==============================================================================
// Project Configuration Flags
// ==============================================================================

FC_DEFINE_EXPORTED_string(FC_HOME_PATH, "/test_ck_yanxishi/flash_ck", "The home path to the root of the project");

FC_DEFINE_EXPORTED_string(FC_ROCM_PATH, "/opt/rocm", "ROCm installation path");

FC_DEFINE_EXPORTED_string(FC_TUNING_DB_DIR, "", "Directory for profiler cache storage");

FC_DEFINE_EXPORTED_string(FC_BUILD_CACHE_DIR, "", "Directory for build cache storage");

FC_DEFINE_EXPORTED_string(FC_COMPILER_OPT_LEVEL, "o3", "Compiler optimization level (o0, o1, o2, o3)");

// ==============================================================================
// Debug and Development Kernel Flags
// ==============================================================================

FC_DEFINE_EXPORTED_bool(FC_DEBUG_KERNEL_INSTANCE, false, "Enable kernel instance debugging");

FC_DEFINE_EXPORTED_bool(FC_SAVE_TEMP_FILE, false, "Save temporary files for debugging and profiling");

FC_DEFINE_EXPORTED_bool(FC_PRINT_KERNEL_SOURCE_USAGE, false, "Print kernel source usage information");

FC_DEFINE_EXPORTED_bool(FC_TIME_COMPILATION, false, "Measure and report compilation time for each make command");

// ==============================================================================
// Compute and Math Flags
// ==============================================================================

FC_DEFINE_EXPORTED_bool(FC_FLUSH_DENORMALS, false, "Flush denormal floating-point numbers to zero");

FC_DEFINE_EXPORTED_bool(FC_USE_FAST_MATH, true, "Enable fast math optimizations (may reduce precision)");

// ==============================================================================
// Build System Flags
// ==============================================================================

FC_DEFINE_EXPORTED_int32(FC_NUM_BUILDERS, -1, "Number of parallel builders (-1 for auto-detection)");

FC_DEFINE_EXPORTED_int32(FC_BUILDING_MAX_ATTEMPTS, 3, "Maximum retry attempts for failed builds");

FC_DEFINE_EXPORTED_int32(FC_BUILDING_TIMEOUT, 300, "Build timeout in seconds");

FC_DEFINE_EXPORTED_bool(FC_COMBINE_PROFILING_SOURCES, false, "Combine multiple profiler sources per target");

// ==============================================================================
// Profiling and Tuning Flags
// ==============================================================================

FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILING_DB, true, "Force the profiler to use cached results");

FC_DEFINE_EXPORTED_bool(FC_FORCE_PROFILING, true, "Force profiling even if cached results exist");

FC_DEFINE_EXPORTED_bool(FC_FLUSH_PROFILING_DB, false, "Flush profile cache database on startup");

FC_DEFINE_EXPORTED_int32(FC_TUNING_MODE, 1, "Tuning mode: 0=heuristic, 1=autotuning, 2=hybrid");

FC_DEFINE_EXPORTED_int32(FC_TUNING_METRIC, 0, "Profiling metric: 0=latency, 1=throughput(TFLOPS), 2=bandwidth");

FC_DEFINE_EXPORTED_int32(FC_TUNING_NUM_COLD_ITERATION, 16, "Number of warm-up iterations for profiling");

FC_DEFINE_EXPORTED_int32(FC_TUNING_NUM_REPEATS, 64, "Number of profiling iterations for measurements");

FC_DEFINE_EXPORTED_bool(FC_TUNING_GPU_TIMER, true, "Enable GPU timer for accurate profiling");

// FC_DEFINE_EXPORTED_bool(FC_TUNING_VERIFY, false, "Enable profiling result verification");

FC_DEFINE_EXPORTED_bool(FC_TUNING_LOG, false, "Enable detailed tuning logs");

FC_DEFINE_EXPORTED_bool(FC_TUNING_FLUSH_CACHE, true, "Flush cache before profiling measurements");

FC_DEFINE_EXPORTED_int32(FC_TUNING_ROTATING_COUNT, 1, "Number of rotating profiling runs");

// ==============================================================================
// Error Handling and Debugging Flags
// ==============================================================================

FC_DEFINE_EXPORTED_int32(FC_CALL_STACK_LEVEL, 1, "Error call stack detail level: 0=summary, 1=+Python, 2=+C++");
