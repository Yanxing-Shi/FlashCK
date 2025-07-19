#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "flashck/core/utils/common.h"

namespace flashck {

/**
 * @class Compiler
 * @brief Static utility class for ROCm/HIP kernel compilation
 *
 * Provides static methods for configuring and invoking the ROCm/HIP compiler
 * to build GPU kernels. All methods are stateless and use global configuration
 * flags to determine compilation parameters.
 *
 * Supported compilation targets:
 * - Object files (.o) for intermediate compilation
 * - Shared libraries (.so) for runtime loading
 * - Executables (.exe) for profiling and testing
 */
class Compiler {
public:
    /**
     * @brief Get the installed ROCm version
     * @return ROCm version string, or "Unknown" if not found
     *
     * Reads the ROCm version from the standard installation path.
     * Used for compatibility checking and debugging information.
     */
    static std::string GetROCmVersion();

    /**
     * @brief Get include paths for compilation
     * @param dst_file_ext Target file extension ("o", "so", "exe")
     * @return Vector of absolute include paths
     *
     * Constructs the necessary include paths for ROCm/HIP compilation,
     * including ComposableKernel headers and ROCm system headers.
     * Adds additional paths for executable targets.
     */
    static std::vector<std::filesystem::path> GetIncludePaths(const std::filesystem::path& dst_file_ext);

    /**
     * @brief Get library linking options
     * @param dst_file_ext Target file extension ("o", "so", "exe")
     * @return Vector of library options and flags
     *
     * Generates appropriate library paths and linking flags based on
     * the target type. Includes ROCm/HIP runtime libraries and
     * additional system libraries for executables.
     */
    static std::vector<std::string> GetLibraryOptions(const std::string& dst_file_ext);

    /**
     * @brief Get compiler optimization and feature flags
     * @return Vector of compiler options
     *
     * Constructs compiler flags based on global configuration:
     * - Optimization level (FC_COMPILER_OPT_LEVEL)
     * - Debug information (FC_DEBUG_KERNEL_INSTANCE)
     * - Temporary file saving (FC_SAVE_TEMP_FILE)
     * - Resource usage analysis (FC_PRINT_KERNEL_SOURCE_USAGE)
     * - Math optimizations (FC_FLUSH_DENORMALS, FC_USE_FAST_MATH)
     */
    static std::vector<std::string> GetCompilerOptions();

    /**
     * @brief Get the path to the ROCm compiler executable
     * @return Absolute path to the compiler (hipcc or clang)
     *
     * Determines the appropriate compiler path based on configuration:
     * - Uses FC_ROCM_PATH if specified
     * - Falls back to default /opt/rocm installation
     */
    static std::filesystem::path GetROCmCompilerPath();

    /**
     * @brief Generate complete compiler command
     * @param src_files Vector of source file paths
     * @param dst_file Output file path
     * @param dst_file_ext Output file extension ("o", "so", "exe")
     * @param extra_args Additional compiler arguments (optional)
     * @return Complete compiler command string
     *
     * Assembles a complete compiler command by combining:
     * - Compiler path and options
     * - Include paths
     * - Library options
     * - Source and output files
     * - Target-specific flags
     *
     * @throws Unimplemented if dst_file_ext is not supported
     */
    static std::string GetCompilerCommand(const std::vector<std::string>& src_files,
                                          const std::string&              dst_file,
                                          const std::string&              dst_file_ext,
                                          const std::vector<std::string>& extra_args = {});
};

}  // namespace flashck