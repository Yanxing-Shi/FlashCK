#include "core/profiling/compiler.h"

// Flag declarations for compiler configuration
FC_DECLARE_string(FC_ROCM_PATH);  ///< ROCm installation path
FC_DECLARE_string(FC_HOME_PATH);  ///< FlashCK home directory path

FC_DECLARE_string(FC_COMPILER_OPT_LEVEL);       ///< Compiler optimization level (O0, O1, O2, O3)
FC_DECLARE_bool(FC_DEBUG_KERNEL_INSTANCE);      ///< Enable debug information and logging
FC_DECLARE_bool(FC_SAVE_TEMP_FILE);             ///< Save intermediate compilation files
FC_DECLARE_bool(FC_PRINT_KERNEL_SOURCE_USAGE);  ///< Print kernel resource usage analysis
FC_DECLARE_bool(FC_FLUSH_DENORMALS);            ///< Flush denormal numbers to zero for performance
FC_DECLARE_bool(FC_USE_FAST_MATH);              ///< Enable fast math optimizations

namespace flashck {

std::string Compiler::GetROCmVersion()
{
    // Standard ROCm version file location
    static const std::string version_file_path = "/opt/rocm/.info/version";

    std::ifstream version_file(version_file_path);
    if (version_file.is_open()) {
        std::string version;
        std::getline(version_file, version);

        // Trim whitespace from version string
        version.erase(0, version.find_first_not_of(" \t\r\n"));
        version.erase(version.find_last_not_of(" \t\r\n") + 1);

        if (!version.empty()) {
            VLOG(2) << "Found ROCm version: " << version;
            return version;
        }
    }

    VLOG(1) << "ROCm version file not found or empty at: " << version_file_path;
    return "Unknown";
}

std::vector<std::filesystem::path> Compiler::GetIncludePaths(const std::filesystem::path& dst_file_ext)
{
    // Base paths for ROCm and ComposableKernel includes
    const std::filesystem::path rocm_include = std::filesystem::path(FLAGS_FC_ROCM_PATH) / "include";
    const std::filesystem::path ck_path      = std::filesystem::path(FLAGS_FC_HOME_PATH) / "3rdparty/composable_kernel";
    const std::filesystem::path ck_include   = ck_path / "include";
    const std::filesystem::path ck_library_include = ck_path / "library" / "include";

    // Essential include paths for all compilation targets
    std::vector<std::filesystem::path> paths = {
        std::filesystem::absolute(ck_include),          // ComposableKernel main headers
        std::filesystem::absolute(ck_library_include),  // ComposableKernel library headers
        std::filesystem::absolute(rocm_include)         // ROCm system headers
    };

    // Additional includes for executable targets (profiling binaries)
    if (dst_file_ext == "exe") {
        std::filesystem::path ck_utility_include = ck_path / "library" / "src" / "utility";
        paths.push_back(std::filesystem::absolute(ck_utility_include));
        VLOG(2) << "Added utility includes for executable target";
    }

    // Log include paths for debugging
    VLOG(2) << "Include paths for target '" << dst_file_ext << "':";
    for (const auto& path : paths) {
        VLOG(2) << "  " << path.string();
    }

    return paths;
}

std::vector<std::string> Compiler::GetLibraryOptions(const std::string& dst_file_ext)
{
    // ROCm library directories
    std::filesystem::path rocm_lib_dir = std::filesystem::path(FLAGS_FC_ROCM_PATH) / "lib";
    std::filesystem::path hip_lib_dir  = std::filesystem::path(FLAGS_FC_ROCM_PATH) / "hip" / "lib";

    // Base library options for all targets
    std::vector<std::string> opts = {
        "-include __clang_hip_runtime_wrapper.h",                 // HIP runtime wrapper
        "-L" + std::filesystem::absolute(rocm_lib_dir).string(),  // ROCm library path
        "-L" + std::filesystem::absolute(hip_lib_dir).string(),   // HIP library path
        "-lamdhip64"                                              // AMD HIP runtime library
    };

    // Additional libraries for executable targets
    if (dst_file_ext == "exe") {
        opts.push_back("-lpthread");  // POSIX threads for profiling utilities
        opts.push_back("-lstdc++");   // Standard C++ library
        VLOG(2) << "Added executable-specific libraries (pthread, stdc++)";
    }

    VLOG(2) << "Library options for target '" << dst_file_ext << "':";
    for (const auto& opt : opts) {
        VLOG(2) << "  " << opt;
    }

    return opts;
}

std::vector<std::string> Compiler::GetCompilerOptions()
{
    // Base compiler options for GPU kernel compilation
    std::vector<std::string> opts = {
        "-" + ToString(FLAGS_FC_COMPILER_OPT_LEVEL),  // Optimization level (O0, O1, O2, O3)
        "-std=c++20",                                 // C++20 standard
        "-fno-gpu-rdc",                               // Disable GPU relocatable device code
        "-fPIC",                                      // Position independent code
        "-fvisibility=hidden",                        // Hidden symbol visibility by default

        // AMDGPU-specific optimizations
        "-mllvm",
        "-amdgpu-early-inline-all=true",  // Aggressive early inlining
        "-mllvm",
        "-amdgpu-function-calls=false",  // Disable function calls in kernels
        "-mllvm",
        "-enable-post-misched=0",  // Disable post-machine instruction scheduling

        "--offload-arch=" + GetDeviceName(0)  // Target GPU architecture
    };

    // Debug options - adds debug symbols and logging
    if (FLAGS_FC_DEBUG_KERNEL_INSTANCE) {
        opts.push_back("-DDEBUG_LOG=1");  // Enable debug logging macros
        opts.push_back("-g");             // Generate debug information
        VLOG(2) << "Debug mode enabled: added debug flags";
    }

    // Development option - save intermediate files for inspection
    if (FLAGS_FC_SAVE_TEMP_FILE) {
        opts.push_back("--save-temps=obj");  // Save temporary files in object directory
        VLOG(2) << "Temporary file saving enabled";
    }

    // Profiling option - report kernel resource usage
    if (FLAGS_FC_PRINT_KERNEL_SOURCE_USAGE) {
        opts.push_back("-Rpass-analysis=kernel-resource-usage");  // Kernel resource analysis
        VLOG(2) << "Kernel resource usage analysis enabled";
    }

    // Performance optimization - flush denormal numbers to zero
    if (FLAGS_FC_FLUSH_DENORMALS) {
        opts.push_back("-fgpu-flush-denormals-to-zero");  // Flush denormals for performance
        VLOG(2) << "Denormal flushing enabled for performance";
    }

    // Performance optimization - enable fast math
    if (FLAGS_FC_USE_FAST_MATH) {
        opts.push_back("-ffast-math");  // Enable aggressive math optimizations
        VLOG(2) << "Fast math optimizations enabled";
    }

    VLOG(2) << "Compiler options configured with " << opts.size() << " flags";
    return opts;
}

std::filesystem::path Compiler::GetROCmCompilerPath()
{
    std::filesystem::path compiler_path;

    if (FLAGS_FC_ROCM_PATH.empty()) {
        // Default ROCm installation path with clang compiler
        compiler_path = std::filesystem::absolute(std::filesystem::path("/opt/rocm") / "llvm" / "bin" / "clang");
        VLOG(2) << "Using default ROCm compiler path: " << compiler_path.string();
    }
    else {
        // Custom ROCm path with hipcc compiler
        compiler_path = std::filesystem::path(FLAGS_FC_ROCM_PATH) / "bin" / "hipcc";
        VLOG(2) << "Using custom ROCm compiler path: " << compiler_path.string();
    }

    return compiler_path;
}

std::string Compiler::GetCompilerCommand(const std::vector<std::string>& src_files,
                                         const std::string&              dst_file,
                                         const std::string&              dst_file_ext,
                                         const std::vector<std::string>& extra_args)
{
    VLOG(2) << "Generating compiler command for target type: " << dst_file_ext;
    VLOG(2) << "Output file: " << dst_file;
    VLOG(2) << "Source files: " << JoinStrings(src_files, ", ");

    // Gather all compilation components
    std::vector<std::filesystem::path> include_paths    = GetIncludePaths(dst_file_ext);
    std::vector<std::string>           lib_options      = GetLibraryOptions(dst_file_ext);
    std::vector<std::string>           compiler_options = GetCompilerOptions();
    std::filesystem::path              compiler_path    = GetROCmCompilerPath();

    // Build command options in order
    std::vector<std::string> options;
    options.push_back(compiler_path.string());
    options.insert(options.end(), compiler_options.begin(), compiler_options.end());

    // Add include paths with -I prefix
    for (const auto& path : include_paths) {
        options.push_back("-I" + path.string());
    }

    // Add library options
    options.insert(options.end(), lib_options.begin(), lib_options.end());

    // Add target-specific compilation flags
    if (dst_file_ext == "o") {
        options.push_back("-x hip -c");  // Compile only, treat as HIP source
    }
    else if (dst_file_ext == "so") {
        options.push_back("-shared");  // Create shared library
    }
    else if (dst_file_ext == "exe") {
        options.push_back("-x hip");  // Link executable, treat as HIP source
    }
    else {
        FC_THROW(Unimplemented("Unsupported output file suffix: {}", dst_file_ext));
    }

    // Add extra arguments if provided
    if (!extra_args.empty()) {
        options.insert(options.end(), extra_args.begin(), extra_args.end());
        VLOG(2) << "Added extra arguments: " << JoinStrings(extra_args, ", ");
    }

    // Add output file specification
    options.push_back("-o");
    options.push_back(dst_file);

    // Add source files
    for (const auto& src_file : src_files) {
        options.push_back(src_file);
    }

    std::string command = JoinStrings(options, " ");
    VLOG(2) << "Generated compiler command (" << options.size() << " arguments)";

    return command;
}

}  // namespace flashck
