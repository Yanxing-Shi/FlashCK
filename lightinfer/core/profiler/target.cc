#include "lightinfer/core/profiler/target.h"

#include <sstream>

#include "lightinfer/core/utils/dtype.h"
#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/file_utils.h"
#include "lightinfer/core/utils/flags.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/printf.h"
#include "lightinfer/core/utils/rocm_info.h"
#include "lightinfer/core/utils/string_utils.h"

LI_DECLARE_string(LI_ROCM_PATH);
LI_DECLARE_string(LI_HOME_PATH);
LI_DECLARE_string(LI_COMPILER_OPT_LEVEL);
LI_DECLARE_string(LI_PROFILER_CACHE_DIR);
LI_DECLARE_bool(LI_FLUSH_PROFILE_CACHE);

namespace lightinfer {

Target::Target():
    ck_root_path_(std::filesystem::path(FLAGS_LI_HOME_PATH) / "3rdparty/composable_kernel"),
    device_name_(GetTargetDeviceName())
{
    LOG(INFO) << "Init Target on device " << device_name_;
    LoadProfileCache();
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
void Target::GenerateKernel(const GenOperationKind&                                                      op_kind,
                            const std::variant<GemmProblem, NormProblem, EmbeddingProblem, FmhaProblem>& problem)
{
    GenerateCKKernelDispatch(op_kind, problem);

    if (op_kind == GenOperationKind::Gemm) {
        target_gemm_kernel_instance_map_ = Emitters::GetInstance()->GetGemmKernelInstanceMap();
    }
    else if (op_kind == GenOperationKind::Norm) {
        target_norm_kernel_instance_map_ = Emitters::GetInstance()->GetNormKernelInstanceMap();
    }
    else if (op_kind == GenOperationKind::Embedding) {
        target_embedding_kernel_instance_map_ = Emitters::GetInstance()->GetEmbeddingKernelInstanceMap();
    }
    else if (op_kind == GenOperationKind::Fmha) {
        if (std::get<FmhaProblem>(problem).operation_kind_ == FmhaOperationKind::Fwd) {
            target_fmha_fwd_kernel_instance_map_ = Emitters::GetInstance()->GetFmhaFwdKernelInstanceMap();
        }
        else if (std::get<FmhaProblem>(problem).operation_kind_ == FmhaOperationKind::FwdAppendKV) {
            target_fmha_fwd_kernel_instance_map_ = Emitters::GetInstance()->GetFmhaFwdKernelInstanceMap();
        }
        else if (std::get<FmhaProblem>(problem).operation_kind_ == FmhaOperationKind::FwdSplitKV) {
            target_fmha_fwd_kernel_instance_map_ = Emitters::GetInstance()->GetFmhaFwdKernelInstanceMap();
        }
        else if (std::get<FmhaProblem>(problem).operation_kind_ == FmhaOperationKind::FwdSplitKVCombine) {
            target_fmha_fwd_kernel_instance_map_ = Emitters::GetInstance()->GetFmhaFwdKernelInstanceMap();
        }
        else {
            LI_THROW(Unimplemented("not implement"));
        }
    }
    else {
        LI_THROW(Unimplemented("not implement"));
    }
}

/*-------------------------------compile----------------------*/
std::vector<std::string> Target::GetCKPaths()
{
    std::vector<std::string> ck_include_path = {
        ck_root_path_, ck_root_path_ / "include/", ck_root_path_ / "library/include/"};

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
        FLAGS_LI_COMPILER_OPT_LEVEL, "-fvisibility=hidden", "-fPIC", "-std=c++17", "-w"};

    if (device_name_ == "gfx908") {
        options.push_back("-DCK_AMD_GPU_GFX908");
        options.push_back("--offload-arch=gfx908");
    }
    else if (device_name_ == "gfx90a") {
        options.push_back("-DCK_AMD_GPU_GFX90A");
        options.push_back("--offload-arch=gfx90a");
    }
    else {
        LI_THROW(Unavailable("Unsupported GPU Arch {}", device_name_));
    }

    for (const auto& path : ck_include_path) {
        options.push_back("-I" + path);
    }

    options.push_back("-lpthread");

    if (is_profile) {
        // need to fix
        options.push_back("-L/torch6.0_rocm6.0_yanxishi/composable_kernel/build/lib/ -l:libutility.a");
    }

    return JoinToString(options, " ");
}

std::string Target::CompileCmd(const std::string& target, const std::string& src, bool enable_execute, bool is_profile)
{

    if (enable_execute) {
        return Sprintf("hipcc {} -o {} {}", BuildComipleOptions(is_profile), target, src);
    }
    else {
        return Sprintf("hipcc {} -x hip -c -o {} {}", BuildComipleOptions(is_profile), target, src);
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
    profile_cache_ptr_ = std::make_shared<ProfileCacheDB>(device_name_, cache_file_path_);
}

// Get local profile cache path for this target.
std::filesystem::path Target::GetProfileCachePath()
{
    std::string           cache_dir       = FLAGS_LI_PROFILER_CACHE_DIR;
    std::string           cache_file_name = Sprintf("lightinfer{}.db", device_name_);
    std::filesystem::path prefix_path = cache_dir.empty() ? std::filesystem::path(FLAGS_LI_HOME_PATH) / ".lightinfer" :
                                                            std::filesystem::path(cache_dir);

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
            prefix_path = CreateTemporaryDirectory(".lightinfer");
            LOG(WARNING) << "mkdir at " << prefix_path.string() << " instead ";
        }
    }

    std::filesystem::path cache_path = prefix_path / cache_file_name;
    if (FLAGS_LI_FLUSH_PROFILE_CACHE) {
        LOG(INFO) << "Flush cache " << cache_path.string();
        std::filesystem::remove_all(cache_path);
    }

    return cache_path;
}

std::tuple<std::string, int64_t> Target::QueryProfileCache(
    const GenOperationKind&                                                                  op_kind,
    const std::variant<EmbeddingQueryEntry, GemmQueryEntry, NormQueryEntry, FmhaQueryEntry>& query)
{
    if (op_kind == GenOperationKind::Gemm) {
        return profile_cache_ptr_->QueryGemm(std::get<GemmQueryEntry>(query));
    }
    else if (op_kind == GenOperationKind::Norm) {
        return profile_cache_ptr_->QueryNorm(std::get<NormQueryEntry>(query));
    }
    else if (op_kind == GenOperationKind::Embedding) {
        return profile_cache_ptr_->QueryEmbedding(std::get<EmbeddingQueryEntry>(query));
    }
    else if (op_kind == GenOperationKind::Fmha) {
        return profile_cache_ptr_->QueryFmha(std::get<FmhaQueryEntry>(query));
    }
    else {
        LI_THROW(Unimplemented("op kind not implement"));
    }
}

void Target::InsertProfileCache(
    const GenOperationKind&                                                                      op_kind,
    const std::variant<EmbeddingRecordEntry, GemmRecordEntry, NormRecordEntry, FmhaRecordEntry>& record)
{
    if (op_kind == GenOperationKind::Gemm) {
        return profile_cache_ptr_->InsertGemm(std::get<GemmRecordEntry>(record));
    }
    else if (op_kind == GenOperationKind::Norm) {
        return profile_cache_ptr_->InsertNorm(std::get<NormRecordEntry>(record));
    }
    else if (op_kind == GenOperationKind::Embedding) {
        return profile_cache_ptr_->InsertEmbedding(std::get<EmbeddingRecordEntry>(record));
    }
    else if (op_kind == GenOperationKind::Fmha) {
        return profile_cache_ptr_->InsertFmha(std::get<FmhaRecordEntry>(record));
    }
    else {
        LI_THROW(Unimplemented("op kind not implement"));
    }
}

/*-----------------------kernel launch----------*/
void Target::DllReader(const std::string& folder_name, const std::string& context_name, const std::string& so_file_name)
{

    std::filesystem::path build_dir = std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / context_name;
    std::filesystem::path so_path   = build_dir / so_file_name;

    if (!CheckWithRetries(so_path, 3, 5)) {
        LI_THROW(Fatal("kernel so file {} is not find", so_path.string()));
    }

    kernel_lib_ = dylib(build_dir.string(), so_file_name, dylib::no_filename_decorations);
}

}  // namespace lightinfer