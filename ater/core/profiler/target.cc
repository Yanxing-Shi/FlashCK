#include "ater/core/profiler/target.h"

#include <sstream>

#include "ater/core/utils/dtype.h"
#include "ater/core/utils/enforce.h"
#include "ater/core/utils/file_utils.h"
#include "ater/core/utils/flags.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/printf.h"
#include "ater/core/utils/rocm_info.h"
#include "ater/core/utils/string_utils.h"

ATER_DECLARE_string(ATER_ROCM_PATH);
ATER_DECLARE_string(ATER_HOME_PATH);
ATER_DECLARE_string(ATER_COMPILER_OPT_LEVEL);
ATER_DECLARE_string(ATER_PROFILER_CACHE_DIR);
ATER_DECLARE_bool(ATER_FLUSH_PROFILE_CACHE);

namespace ater {

Target::Target():
    // ck_root_path_(std::filesystem::current_path().root_directory() / "TensorEngine/3rdparty/composable_kernel"),
    ck_root_path_(std::filesystem::path(FLAGS_ATER_HOME_PATH) / "3rdparty/composable_kernel"),
    device_name_(GetTargetDeviceName())
{
    LOG(INFO) << "Init Target[Device] " << device_name_;
}

Target::~Target()
{
    // delete ck profiler file
    // if (is_clear_file_ && std::fileystem::exists(profile_file_path_)) {
    //     std::filesystem::remove_all(profile_file_path);
    // }
}

Target* Target::Instance()
{
    static Target target_instance = Target();
    return &target_instance;
}

std::vector<int> Target::GetTargetSelectedDevices()
{
    return GetSelectedDevices();
}

std::string Target::GetTargetDeviceArch()
{
    return GetDeviceArch();
}

std::string Target::GetTargetDeviceName()
{
    return GetDeviceName();
}

// Genertate op and return operations map
void Target::GenerateKernel(const GenOperationKind& op_kind, const GemmProblem& gemm_problem)
{
    // load cache
    LoadProfileCache();

    // generate and choose the right ops to launch
    GenerateCKKernelDispatch(op_kind, gemm_problem);
    target_kernel_instance_map_ = Emitters::GetInstance()->GetKernelInstanceMap();
}

/*-------------------------------compile----------------------*/
std::vector<std::string> Target::GetCKPaths()
{
    std::vector<std::string> ck_include_path = {ck_root_path_,
                                                ck_root_path_ / "include/",
                                                ck_root_path_ / "external/include/half/",
                                                ck_root_path_ / "library/include/",
                                                ck_root_path_ / "profiler/include/"};

    return ck_include_path;
}

/*
Build compilation commands, including compilation flag library and includes.

        Returns
        -------
        List
            List of compilation options.

        Raises
        ------
        RuntimeError
            Unsupported GPU Arch.
*/
std::string Target::BuildComipleOptions(bool is_profile)
{
    std::vector<std::string> ck_include_path = GetCKPaths();
    std::vector<std::string> options         = {
        FLAGS_ATER_COMPILER_OPT_LEVEL,
        "-fPIC",
        "-fvisibility=hidden",
        "-std=c++17",
        "-w",
        "-DCK_TIME_KERNEL=0",
        Sprintf("-Xclang -mlink-builtin-bitcode -Xclang {}/amdgcn/bitcode/oclc_abi_version_400.bc",
                FLAGS_ATER_ROCM_PATH)};

    if (device_name_ == "gfx908") {
        options.push_back("-DCK_AMD_GPU_GFX908");
        options.push_back("--offload-arch=gfx908");
    }
    else if (device_name_ == "gfx90a") {
        options.push_back("-DCK_AMD_GPU_GFX90A");
        options.push_back("--offload-arch=gfx90a");
    }
    else {
        ATER_THROW(Unavailable("Unsupported GPU Arch {}", device_name_));
    }

    for (const auto& path : ck_include_path) {
        options.push_back("-I" + path);
    }

    std::filesystem::path rocrand_path = std::filesystem::path(FLAGS_ATER_ROCM_PATH) / "rocrand/lib/";
    if (is_profile) {
        options.push_back("-L" + rocrand_path.string());
        options.push_back("-lrocrand");

        options.push_back("-L/usr/local/lib/");
        options.push_back("-lglog");
    }
    return JoinToString(options, " ");
}

std::string Target::CompileCmd(const std::string& target, const std::string& src, bool enable_execute, bool is_profile)
{
    if (enable_execute) {
        return Sprintf("hipcc {} -o {} {}", BuildComipleOptions(is_profile), target, src);
    }
    else {
        if (!is_profile) {
            return Sprintf(
                "/opt/rocm/llvm/bin/clang++ --cuda-device-only -O3 -std=c++17 --offload-arch=gfx90a:sramecc+:xnack- -I/ater_test_yanxishi/TensorEngine/3rdparty/composable_kernel/include/ -x hip -c -o {} {}",
                target,
                src);
        }
        // return Sprintf("hipcc {} -x hip -c -o {} {}", BuildComipleOptions(is_profile), target, src);
    }
}

/*-----------Profiler cache-------------*/

// Load local profile cache for this device.
void Target::LoadProfileCache()
{
    cache_file_path_ = GetProfileCachePath();
    if (cache_file_path_.string().empty())
        return;
    LOG(INFO) << "Loading profile cache from " << cache_file_path_.string();
    profile_cache_ptr_ = make_shared<ProfileCacheDB>(device_name_, cache_file_path_);
}

// Get local profile cache path for this target.
std::filesystem::path Target::GetProfileCachePath()
{
    std::string           cache_dir       = FLAGS_ATER_PROFILER_CACHE_DIR;
    std::string           cache_file_name = Sprintf("ater_{}.db", device_name_);
    std::filesystem::path prefix_path =
        cache_dir.empty() ? std::filesystem::path(FLAGS_ATER_HOME_PATH) / ".ater" : std::filesystem::path(cache_dir);

    if (std::filesystem::exists(prefix_path)) {
        LOG(WARNING) << "Cache Path: " << prefix_path.string() << " is already existed";
    }
    else {
        bool create_result = std::filesystem::create_directories(prefix_path);
        if (create_result) {
            LOG(INFO) << "Created cache directory: " << prefix_path.string();
        }
        else {
            std::filesystem::permissions(prefix_path, std::filesystem::perms::all);  // permission
            LOG(WARNING) << "Cannot mkdir at " << prefix_path.string();
            prefix_path = CreateTemporaryDirectory(".ater");
            LOG(WARNING) << "mkdir at " << prefix_path.string() << " instead ";
        }
    }

    std::filesystem::path cache_path = prefix_path / cache_file_name;
    if (FLAGS_ATER_FLUSH_PROFILE_CACHE) {
        LOG(INFO) << "Flush cache " << cache_path.string();
        std::filesystem::remove_all(cache_path);
    }

    return cache_path;
}

// Get the current profile cache version for the op_class
int Target::GetProfileCacheVersion(const std::string& op_kind)
{
    if (op_kind == "gemm") {
        return profile_cache_ptr_->GetGemmCacheVersion();
    }
    // else if (op_name == "norm") {
    //     return profile_cache_ptr_->GetNormCacheVersion();
    // }
    else {
        ATER_THROW(Unimplemented("{} is not implement", op_kind));
    }
}

// Query the profile cache for the given op class and args
std::tuple<std::string, int, int>
Target::QueryProfileCache(const std::string&                                                     op_kind,
                          const std::unordered_map<std::string, std::variant<int, std::string>>& args)
{
    if (op_kind == "gemm") {
        return profile_cache_ptr_->QueryGemm(args);
    }
    // else if (op_name == "norm") {
    //     // return profile_cache_ptr_->QueryNorm(args);
    // }
    else {
        ATER_THROW(Unimplemented("{} is not implement", op_kind));
    }
}

// Insert the profile cache for the given op class and args.
void Target::InsertProfileCache(const std::string&                                                     op_kind,
                                const std::unordered_map<std::string, std::variant<int, std::string>>& args)
{
    if (op_kind == "gemm") {
        return profile_cache_ptr_->InsertGemm(args);
    }
    // else if (op_name == "norm") {
    //     // return profile_cache_ptr_->InsertNorm(args);
    // }
    else {
        ATER_THROW(Unimplemented("{} is not implement", op_kind));
    }
}

}  // namespace ater