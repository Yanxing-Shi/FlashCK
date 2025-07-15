#include "flashck/core/profiling/compiler.h"

FC_DECLARE_string(FC_ROCM_PATH);
FC_DECLARE_string(FC_HOME_PATH);

FC_DECLARE_string(FC_COMPILER_OPT_LEVEL);
FC_DECLARE_bool(FC_DEBUG_KERNEL_INSTANCE);
FC_DECLARE_bool(FC_SAVE_TEMP_FILE);
FC_DECLARE_bool(FC_PRINT_KERNEL_SOURCE_USAGE);
FC_DECLARE_bool(FC_FLUSH_DENORMALS);
FC_DECLARE_bool(FC_USE_FAST_MATH);

namespace flashck {

std::string Compiler::GetROCmVersion()
{
    std::ifstream version_file("/opt/rocm/.info/version");
    if (version_file.is_open()) {
        std::string version;
        std::getline(version_file, version);
        return version;
    }
    return "Unknown";
}

std::vector<std::filesystem::path> Compiler::GetIncludePaths(const std::filesystem::path& dst_file_ext)
{
    const std::filesystem::path rocm_include = std::filesystem::path(FLAGS_FC_ROCM_PATH) / "include";
    const std::filesystem::path ck_path      = std::filesystem::path(FLAGS_FC_HOME_PATH) / "3rdparty/composable_kernel";
    const std::filesystem::path ck_include   = ck_path / "include";
    const std::filesystem::path ck_library_include = ck_path / "library" / "include";

    std::vector<std::filesystem::path> paths = {std::filesystem::absolute(ck_include),
                                                std::filesystem::absolute(ck_library_include),
                                                std::filesystem::absolute(rocm_include)};

    if (dst_file_ext == "exe") {
        std::filesystem::path ck_utility_include = (ck_path / "library" / "src" / "utility");
        paths.push_back(std::filesystem::absolute(ck_utility_include));
    }

    return paths;
}

std::vector<std::string> Compiler::GetLibraryOptions(const std::string& dst_file_ext)
{
    std::filesystem::path rocm_lib_dir = std::filesystem::path(FLAGS_FC_ROCM_PATH) / "lib";
    std::filesystem::path hip_lib_dir  = std::filesystem::path(FLAGS_FC_ROCM_PATH) / "hip" / "lib";

    std::vector<std::string> opts = {"-include __clang_hip_runtime_wrapper.h",
                                     "-L" + std::filesystem::absolute(rocm_lib_dir).string(),
                                     "-L" + std::filesystem::absolute(hip_lib_dir).string(),
                                     "-lamdhip64"};

    if (dst_file_ext == "exe") {
        opts.push_back("-lpthread");
        opts.push_back("-lstdc++");
    }
    return opts;
}

std::vector<std::string> Compiler::GetCompilerOptions()
{
    std::vector<std::string> opts = {"-" + ToString(FLAGS_FC_COMPILER_OPT_LEVEL),
                                     "-std=c++17",
                                     "-fno-gpu-rdc",
                                     "-fPIC",
                                     "-fvisibility=hidden",
                                     "-mllvm",
                                     "-amdgpu-early-inline-all=true",
                                     "-mllvm",
                                     "-amdgpu-function-calls=false",
                                     "-mllvm",
                                     "-enable-post-misched=0",
                                     "--offload-arch=" + GetDeviceName(0)};

    if (FLAGS_FC_DEBUG_KERNEL_INSTANCE) {
        opts.push_back("-DDEBUG_LOG=1");
        opts.push_back("-g");
    }
    if (FLAGS_FC_SAVE_TEMP_FILE) {
        opts.push_back("--save-temps=obj");
    }
    if (FLAGS_FC_PRINT_KERNEL_SOURCE_USAGE) {
        opts.push_back("-Rpass-analysis=kernel-resource-usage");
    }
    if (FLAGS_FC_FLUSH_DENORMALS) {
        opts.push_back("-fgpu-flush-denormals-to-zero");
    }
    if (FLAGS_FC_USE_FAST_MATH) {
        opts.push_back("-ffast-math");
    }

    return opts;
}

std::filesystem::path Compiler::GetROCmCompilerPath()
{
    if (FLAGS_FC_ROCM_PATH.empty()) {
        return std::filesystem::absolute(std::filesystem::path("/opt/rocm") / "llvm" / "bin" / "clang");
    }

    return std::filesystem::path(FLAGS_FC_ROCM_PATH) / "bin" / "hipcc";
}

std::string Compiler::GetCompilerCommand(const std::vector<std::string>& src_files,
                                         const std::string&              dst_file,
                                         const std::string&              dst_file_ext,
                                         const std::vector<std::string>& extra_args)
{

    std::vector<std::filesystem::path> include_paths    = GetIncludePaths(dst_file_ext);
    std::vector<std::string>           lib_options      = GetLibraryOptions(dst_file_ext);
    std::vector<std::string>           compiler_options = GetCompilerOptions();
    std::filesystem::path              compiler_path    = GetROCmCompilerPath();

    std::vector<std::string> options;
    options.push_back(compiler_path.string());
    options.insert(options.end(), compiler_options.begin(), compiler_options.end());

    for (const auto& path : include_paths) {
        options.push_back("-I" + path.string());
    }

    options.insert(options.end(), lib_options.begin(), lib_options.end());

    if (dst_file_ext == "o") {
        options.push_back("-x hip -c");
    }
    else if (dst_file_ext == "so") {
        options.push_back("-shared");
    }
    else if (dst_file_ext == "exe") {
        options.push_back("-x hip");
    }
    else {
        FC_THROW(Unimplemented("Unsupported output file suffix: {}", dst_file_ext));
    }

    options.push_back("-o");
    options.push_back(dst_file);
    for (const auto& src_file : src_files) {
        options.push_back(src_file);
    }

    return JoinStrings(options, " ");
}

}  // namespace flashck
