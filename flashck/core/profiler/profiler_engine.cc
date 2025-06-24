#include "flashck/core/profiler/target.h"

#include <sstream>

#include "flashck/core/utils/dtype.h"
#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/log.h"
#include "flashck/core/utils/printf.h"
#include "flashck/core/utils/rocm_info.h"
#include "flashck/core/utils/string_utils.h"

LI_DECLARE_string(LI_ROCM_PATH);
LI_DECLARE_string(LI_HOME_PATH);
LI_DECLARE_string(LI_COMPILER_OPT_LEVEL);
LI_DECLARE_string(LI_PROFILER_CACHE_DIR);
LI_DECLARE_bool(LI_FLUSH_PROFILE_CACHE);
LI_DECLARE_bool(LI_IF_DELETE_PROFILE_FILE);

namespace flashck {

ProfilerEngine::ProfilerEngine():
    ck_root_path_(std::filesystem::path(FLAGS_LI_HOME_PATH) / "3rdparty/composable_kernel"),
    device_name_(GetTargetDeviceName())
{
    // Validate critical environment configuration before initialization
    if (FLAGS_LI_HOME_PATH.empty()) {
        LOG(ERROR) << "LI_HOME_PATH environment variable not configured";
        throw std::runtime_error("Missing LI_HOME_PATH configuration");
    }

    // Log normalized path to ensure clean directory structure display
    LOG(INFO) << "Initializing ProfilerEngine device [" << device_name_
              << "] with CK root: " << ck_root_path_.lexically_normal().string();

    try {
        // Load hardware profile cache for performance optimization
        LoadProfileCache();
    }
    catch (const std::exception& e) {
        // Convert storage errors to fatal initialization failures
        LOG(ERROR) << "Profile cache initialization failed: " << e.what();
        throw;  // Propagate exception to prevent invalid object creation
    }
}

ProfilerEngine::~ProfilerEngine()
{
    // Early exit if profile cleanup is disabled via feature flag
    if (!LI_IF_DELETE_PROFILE_FILE)
        return;

    const auto& profile_path = profile_file_path_;

    // Skip cleanup for uninitialized paths
    if (profile_path.empty()) {
        VLOG(3) << "Profile cleanup: No path specified";
        return;
    }

    std::error_code ec;

    // Unified filesystem removal (handles both files and directories)
    const uintmax_t removed_count = std::filesystem::remove_all(profile_path, ec);

    // Error handling for filesystem operations
    if (ec) {
        LOG(ERROR) << "Profile cleanup failed for " << profile_path << " | Error: " << ec.message() << " ("
                   << ec.value() << ")";
    }
    else {
        // Success logging with verbosity control
        VLOG(1) << "Profile cleanup: Removed " << removed_count << " entries from " << profile_path;
    }
}

/// @brief Provides singleton access to the ProfilerEngine instance
/// @return Reference to the thread-safe singleton instance
/// @note Constructor/destructor hidden, copy/move operations deleted
[[nodiscard]] ProfilerEngine& ProfilerEngine::GetInstance() noexcept
{
    static ProfilerEngine instance;  // Direct construction (C++11 thread-safe)
    return instance;
}

/// @brief Generates kernel operations and populates instance maps
/// @param op_kind Operation type to generate
/// @param problem Variant containing specific operation parameters
/// @throws UnimplementedError for unsupported operations
void ProfilerEngine::GenerateKernel(GenOperationKind op_kind, const ProblemVariant& problem)
{
    GenerateCKKernelDispatch(op_kind, problem);

    // Use structured binding for state management
    auto& emitter = *Emitters::GetInstance();

    switch (op_kind) {
        case GenOperationKind::Gemm:
            target_gemm_kernel_instance_map_ = emitter.GetGemmKernelInstanceMap();
            break;

        case GenOperationKind::Norm:
            target_norm_kernel_instance_map_ = emitter.GetNormKernelInstanceMap();
            break;

        case GenOperationKind::Embedding:
            target_embedding_kernel_instance_map_ = emitter.GetEmbeddingKernelInstanceMap();
            break;

        case GenOperationKind::Fmha: {
            // Type-safe variant access
            if (!std::holds_alternative<FmhaProblem>(problem)) {
                LI_THROW(TypeError, "FMHA operation requires FmhaProblem variant");
            }

            const auto& fmha_prob = std::get<FmhaProblem>(problem);
            if (IsForwardOperation(fmha_prob.operation_kind_)) {
                target_fmha_fwd_kernel_instance_map_ = emitter.GetFmhaFwdKernelInstanceMap();
            }
            else {
                LI_THROW(UnimplementedError,
                         "Non-forward FMHA operations not implemented. Requested operation: {}",
                         static_cast<int>(fmha_prob.operation_kind_));
            }
            break;
        }

        default:
            LI_THROW(UnimplementedError, "Unsupported operation kind: {}", static_cast<int>(op_kind));
    }
}

// Helper function for FMHA operation validation
bool ProfilerEngine::IsForwardOperation(FmhaOperationKind kind) const noexcept
{
    return kind == FmhaOperationKind::Fwd || kind == FmhaOperationKind::FwdAppendKV
           || kind == FmhaOperationKind::FwdSplitKV || kind == FmhaOperationKind::FwdSplitKVCombine;
}

/// @brief Retrieves currently selected device IDs
/// @return Copy of device ID list (vector guaranteed NRVO)
std::vector<int> ProfilerEngine::GetTargetSelectedDevices() const
{
    return GetSelectedDevices();
}

/// @brief Gets architecture identifier for target device
/// @return Architecture name as string
std::string ProfilerEngine::GetTargetDeviceArch() const
{
    return GetDeviceArch();
}

/// @brief Retrieves human-readable device name
/// @return Device marketing name string
std::string ProfilerEngine::GetTargetDeviceName() const
{
    return GetDeviceName();
}

/*-------------------------------compile----------------------*/
/// @brief Retrieves Composable Kernel include paths for compiler configuration
/// @return Vector of include paths formatted as strings
/// @note Ensures proper path concatenation using filesystem semantics
std::vector<std::string> ProfilerEngine::GetCKPaths() const
{
    // Validate base path before concatenation
    if (ck_root_path_.empty()) {
        LOG(WARNING) << "CK root path is empty - verify library configuration";
    }

    return {
        ck_root_path_.string(), (ck_root_path_ / "include").string(), (ck_root_path_ / "library" / "include").string()};
}

/// @brief Generates compiler options for HIP compilation targeting specific AMD architectures
/// @param is_profile Whether to include profiling library dependencies
/// @return Space-separated compiler options string
/// @throws UnavailableError for unsupported GPU architectures
std::string ProfilerEngine::GenerateCompileOptions(bool is_profile) const
{
    // Architecture-specific configuration
    static constexpr std::array kSupportedArchitectures = {
        std::pair{"gfx908", std::pair{"-DCK_AMD_GPU_GFX908", "--offload-arch=gfx908"}},
        std::pair{"gfx90a", std::pair{"-DCK_AMD_GPU_GFX90A", "--offload-arch=gfx90a"}}};

    const std::vector<std::string> base_options = {
        FLAGS_LI_COMPILER_OPT_LEVEL, "-fvisibility=hidden", "-fPIC", "-std=c++17", "-w"};

    std::vector<std::string> options(base_options);
    options.reserve(base_options.size() + 6);  // Anticipate max additions

    // Add architecture-specific flags
    const auto add_arch_flags = [&](const auto& arch) {
        options.push_back(arch.second.first);
        options.push_back(arch.second.second);
    };

    const auto arch_it = std::find_if(kSupportedArchitectures.begin(),
                                      kSupportedArchitectures.end(),
                                      [this](const auto& arch) { return arch.first == device_name_; });

    if (arch_it == kSupportedArchitectures.end()) {
        LI_THROW(UnavailableError, "Unsupported GPU Architecture: {}", device_name_);
    }
    add_arch_flags(*arch_it);

    // Add include paths
    const auto ck_include_paths = GetCKPaths();
    for (const auto& path : ck_include_paths) {
        options.push_back(absl::StrCat("-I", path));
    }

    options.push_back("-lpthread");

    // Profile-specific configuration
    if (is_profile) {
        static constexpr std::string_view kProfileLibPath = "/torch6.0_rocm6.0_yanxishi/composable_kernel/build/lib/";
        static constexpr std::string_view kProfileLib     = "libutility.a";

        options.push_back(absl::StrCat("-L", kProfileLibPath));
        options.push_back(absl::StrCat("-l:", kProfileLib));
    }

    return absl::StrJoin(options, " ");
}

/// @brief Constructs HIP compilation command for the target
/// @param target Output file path
/// @param src Source file path
/// @param enable_execute Whether to build executable (true) or object (false)
/// @param is_profile Whether to include profiling instrumentation
/// @return Full compilation command string
std::string ProfilerEngine::BuildCompilationCommand(std::string_view target,
                                                    std::string_view src,
                                                    bool             enable_execute,
                                                    bool             is_profile) const
{
    const std::string_view compile_mode = enable_execute ? "-o" : "-x hip -c -o";

    return absl::StrFormat("hipcc %s %s %s %s", GenerateCompileOptions(is_profile), compile_mode, target, src);
}

/// @brief Loads profile cache database for the target device
/// @details Attempts to load from configured path, creates fallback directory if needed
/// @throws FatalError if critical filesystem operations fail
void ProfilerEngine::LoadProfileCache()
{
    cache_file_path_ = GetProfileCachePath();
    if (cache_file_path_.empty()) {
        LOG(WARNING) << "Profile cache disabled: No valid cache path available";
        return;
    }

    try {
        profile_cache_ptr_ = std::make_shared<ProfileCacheDB>(device_name_, cache_file_path_);
        LOG(INFO) << "Initialized profile cache at: " << cache_file_path_.string();
    }
    catch (const std::exception& e) {
        LI_THROW(FatalError, "Failed to initialize profile cache: {}", e.what());
    }
}

/// @brief Constructs validated cache path following hierarchy:
///        1. Explicit config flag path
///        2. LI_HOME-based default path
///        3. Temporary fallback directory
/// @return Validated cache file path or empty path if all options fail
std::filesystem::path ProfilerEngine::GetProfileCachePath()
{
    // Configure base paths
    const auto [base_path, source] = GetBaseCachePath();
    if (base_path.empty()) {
        LOG(ERROR) << "No valid base path available for profile cache";
        return {};
    }

    // Ensure directory existence
    if (!EnsureDirectoryExists(base_path)) {
        LOG(WARNING) << "Failed to create primary cache directory: " << base_path.string();
        return TryFallbackCachePath();
    }

    // Construct final cache path
    const std::filesystem::path cache_path = base_path / FormatCacheFileName();

    // Handle cache flushing
    if (FLAGS_LI_FLUSH_PROFILE_CACHE) {
        FlushExistingCache(cache_path);
    }

    return cache_path;
}

// Helper functions decomposed from GetProfileCachePath
namespace {
/// @brief Returns prioritized base path and configuration source
std::tuple<std::filesystem::path, std::string> GetBaseCachePath()
{
    if (!FLAGS_LI_PROFILER_CACHE_DIR.empty()) {
        return {FLAGS_LI_PROFILER_CACHE_DIR, "config flag"};
    }

    const std::filesystem::path default_path = std::filesystem::path(FLAGS_LI_HOME_PATH) / ".flashck";
    return {default_path, "default LI_HOME location"};
}

/// @brief Creates directory with full error diagnostics
bool EnsureDirectoryExists(const std::filesystem::path& path)
{
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        LOG(INFO) << "Using existing cache directory: " << path.string();
        return true;
    }

    if (std::filesystem::create_directories(path, ec)) {
        LOG(INFO) << "Created cache directory: " << path.string();
        return true;
    }

    LOG(ERROR) << "Directory creation failed for " << path.string() << ": " << ec.message();
    LogFilesystemSpace(path);
    return false;
}

/// @brief Attempts temporary directory fallback
std::filesystem::path TryFallbackCachePath()
{
    std::error_code             ec;
    const std::filesystem::path temp_dir = CreateTemporaryDirectory("flashck_fallback");

    if (temp_dir.empty() || !std::filesystem::exists(temp_dir, ec)) {
        LOG(QFATAL) << "Critical failure: Cannot create fallback cache directory";
        return {};
    }

    LOG(WARNING) << "Using fallback cache location: " << temp_dir.string();
    return temp_dir / FormatCacheFileName();
}

/// @brief Generates standardized cache filename
std::string FormatCacheFileName()
{
    return absl::StrFormat("flashck_%s.db", device_name_);
}

/// @brief Removes existing cache files with validation
void FlushExistingCache(const std::filesystem::path& path)
{
    std::error_code ec;
    if (std::filesystem::remove(path, ec)) {
        LOG(INFO) << "Successfully flushed cache: " << path.string();
        return;
    }

    LOG(WARNING) << "Cache flush failed for " << path.string() << ": " << (ec ? ec.message() : "Unknown error");
}

/// @brief Logs filesystem space information for diagnostics
void LogFilesystemSpace(const std::filesystem::path& path)
{
    std::error_code                   ec;
    const std::filesystem::space_info space = std::filesystem::space(path, ec);

    if (!ec) {
        LOG(INFO) << "Filesystem space at " << path.string() << ": "
                  << "Available=" << space.available << " bytes, "
                  << "Capacity=" << space.capacity << " bytes";
    }
}

std::tuple<std::string, int64_t> ProfilerEngine::QueryProfileCache(
    GenOperationKind                                                                         op_kind,
    const std::variant<EmbeddingQueryEntry, GemmQueryEntry, NormQueryEntry, FmhaQueryEntry>& query)
{
    // Static assertion ensures variant type count matches case branches
    static_assert(std::variant_size_v<decltype(query)> == 4, "Update query dispatch when adding new operation types");

    // Type-safe dispatch based on operation category
    switch (op_kind) {
        case GenOperationKind::Gemm:
            return profile_cache_ptr_->QueryGemm(std::get<GemmQueryEntry>(query));
        case GenOperationKind::Norm:
            return profile_cache_ptr_->QueryNorm(std::get<NormQueryEntry>(query));
        case GenOperationKind::Embedding:
            return profile_cache_ptr_->QueryEmbedding(std::get<EmbeddingQueryEntry>(query));
        case GenOperationKind::Fmha:
            return profile_cache_ptr_->QueryFmha(std::get<FmhaQueryEntry>(query));
        default:
            LI_THROW(UnimplementedError,
                     "Unsupported operation kind: %d",
                     static_cast<std::underlying_type_t<GenOperationKind>>(op_kind));
    }
}

void ProfilerEngine::InsertProfileCache(
    GenOperationKind                                                                             op_kind,
    const std::variant<EmbeddingRecordEntry, GemmRecordEntry, NormRecordEntry, FmhaRecordEntry>& record)
{
    // Dispatch to specialized cache insertion based on operation type
    switch (op_kind) {
        case GenOperationKind::Gemm:
            profile_cache_ptr_->InsertGemm(std::get<GemmRecordEntry>(record));
            return;
        case GenOperationKind::Norm:
            profile_cache_ptr_->InsertNorm(std::get<NormRecordEntry>(record));
            return;
        case GenOperationKind::Embedding:
            profile_cache_ptr_->InsertEmbedding(std::get<EmbeddingRecordEntry>(record));
            return;
        case GenOperationKind::Fmha:
            profile_cache_ptr_->InsertFmha(std::get<FmhaRecordEntry>(record));
            return;
        default:
            // Handle future expansion during compilation phase
            static_assert(std::variant_size_v<decltype(record)> == 4,
                          "Update switch statement when adding new operation record types");
    }

    // Runtime fallback for invalid operation types
    LI_THROW(UnimplementedError,
             "Unsupported operation kind: %d",
             static_cast<std::underlying_type_t<GenOperationKind>>(op_kind));
}

void ProfilerEngine::LoadKernelLibrary(std::string_view folder_name,
                                       std::string_view context_name,
                                       std::string_view so_file_name)
{
    // Validate environment configuration
    if (FLAGS_LI_HOME_PATH.empty()) {
        LI_THROW(Fatal("LI_HOME environment path not configured"));
    }

    // Construct full library path
    const std::filesystem::path lib_path =
        std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / context_name / so_file_name;

    // Verify library existence
    if (!std::filesystem::exists(lib_path)) {
        LI_THROW(Fatal("Kernel library not found at {}", lib_path.string()));
    }

    // Attempt dynamic loading
    try {
        kernel_lib_ =
            dylib(lib_path.parent_path().string(), lib_path.filename().string(), dylib::no_filename_decorations);
    }
    catch (const dylib::load_error& e) {
        LI_THROW(Fatal("Failed to load {}: {}", lib_path.string(), e.what()));
    }
}

}  // namespace flashck