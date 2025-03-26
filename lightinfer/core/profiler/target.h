#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "lightinfer/core/profiler/emitters.h"
#include "lightinfer/core/profiler/generator.h"
#include "lightinfer/core/profiler/profile_cache.h"

#include "lightinfer/core/utils/dylib_utils.h"

namespace lightinfer {

class Target {
public:
    explicit Target();

    ~Target();

    static Target* Instance();

    // device info and property utility
    std::vector<int> GetTargetSelectedDevices();
    std::string      GetTargetDeviceArch();
    std::string      GetTargetDeviceName();

    // Genertate kernel and return kernel map
    void GenerateKernel(const GenOperationKind&                                                      op_kind,
                        const std::variant<GemmProblem, NormProblem, EmbeddingProblem, FmhaProblem>& problem);

    /*----------------compile-----------------*/
    std::vector<std::string> GetCKPaths();
    std::filesystem::path    GetROCmPkgPath();
    std::string              BuildComipleOptions(bool is_profile = true);
    std::string
    CompileCmd(const std::string& target, const std::string& src, bool enable_execute = false, bool is_profile = false);

    /*-----------Profiler cache-------------*/
    std::filesystem::path GetProfileCachePath();
    void                  LoadProfileCache();
    int                   GetProfileCacheVersion(const GenOperationKind& op_kind);
    std::tuple<std::string, int64_t>
         QueryProfileCache(const GenOperationKind&                                                                  op_kind,
                           const std::variant<EmbeddingQueryEntry, GemmQueryEntry, NormQueryEntry, FmhaQueryEntry>& query);
    void InsertProfileCache(
        const GenOperationKind&                                                                      op_kind,
        const std::variant<EmbeddingRecordEntry, GemmRecordEntry, NormRecordEntry, FmhaRecordEntry>& record);

    std::map<GemmOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_gemm_kernel_instance_map_;
    std::map<NormOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_norm_kernel_instance_map_;
    std::map<EmbeddingOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_embedding_kernel_instance_map_;
    std::map<FmhaOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_fmha_fwd_kernel_instance_map_;

    /*-----------------------kernel launch----------*/
    void DllReader(const std::string& folder_name,
                   const std::string& context_name,
                   const std::string& so_file_name = "generated_kernel.so");

    dylib kernel_lib_;

private:
    std::filesystem::path ck_root_path_;
    std::string           device_name_;

    std::filesystem::path           cache_file_path_;
    std::shared_ptr<ProfileCacheDB> profile_cache_ptr_;
};

}  // namespace lightinfer