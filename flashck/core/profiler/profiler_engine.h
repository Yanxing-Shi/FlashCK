#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "flashck/core/profiler/emitters.h"
#include "flashck/core/profiler/generator.h"
#include "flashck/core/profiler/profile_cache.h"

#include "flashck/core/utils/dylib_utils.h"

namespace flashck {

class ProfilerEngine {
public:
    explicit ProfilerEngine();
    ProfilerEngine(const ProfilerEngine&)            = delete;
    ProfilerEngine& operator=(const ProfilerEngine&) = delete;
    ProfilerEngine(ProfilerEngine&&)                 = delete;
    ProfilerEngine& operator=(ProfilerEngine&&)      = delete;

    ~ProfilerEngine();

    static ProfilerEngine& Instance();

    // device info and property utility
    std::vector<int> GetTargetSelectedDevices() const;
    std::string      GetTargetDeviceArch() const;
    std::string      GetTargetDeviceName() const;

    // Genertate kernel and return kernel map
    void GenerateKernel(const GenOperationKind&                                                      op_kind,
                        const std::variant<GemmProblem, NormProblem, EmbeddingProblem, FmhaProblem>& problem);

    /*----------------compile-----------------*/
    std::vector<std::string> GetCKPaths() const;
    std::filesystem::path    GetROCmPkgPath() const;
    std::string              BuildCompileOptions(bool is_profile = true);
    std::string
    CompileCmd(const std::string& target, const std::string& src, bool enable_execute = false, bool is_profile = false);

    /*-----------Profiler cache-------------*/
    /// @brief Constructs validated cache path following hierarchy:
    ///        1. Explicit config flag path
    ///        2. LI_HOME-based default path
    ///        3. Temporary fallback directory
    /// @return Validated cache file path or empty path if all options fail
    std::filesystem::path GetProfileCachePath();

    void LoadProfileCache();
    int  GetProfileCacheVersion(const GenOperationKind& op_kind);

    /// @brief Queries profile cache for optimized kernel configuration
    /// @param op_kind Operation type category (GEMM/Norm/Embedding/etc)
    /// @param query Variant containing operation-specific query parameters
    /// @return Tuple containing kernel configuration string and split-k value
    /// @throws UnimplementedError for unsupported operation types
    /// @note Maintains strict type safety between op_kind and query variant type
    std::tuple<std::string, int64_t>
    QueryProfileCache(GenOperationKind                                                                         op_kind,
                      const std::variant<EmbeddingQueryEntry, GemmQueryEntry, NormQueryEntry, FmhaQueryEntry>& query);

    /// @brief Inserts a profiling record into the appropriate type-specific cache
    /// @param op_kind Operation category type (GEMM/Norm/Embedding/etc)
    /// @param record Variant containing concrete operation record entry
    /// @throws UnimplementedError for unsupported operation types
    /// @note Enforces type safety between operation kind and record variant type
    void InsertProfileCache(
        GenOperationKind op_kind,  // Pass by value for enum types
        const std::variant<EmbeddingRecordEntry, GemmRecordEntry, NormRecordEntry, FmhaRecordEntry>& record);

    /// @brief Loads a kernel shared library from specified paths
    /// @param folder_name Top-level directory under LI_HOME (e.g., "bin")
    /// @param context_name Context-specific subdirectory (e.g., "cuda_110")
    /// @param so_file_name Shared object filename (e.g., "kernel.so")
    /// @throws FatalError if paths are invalid or library loading fails
    /// @note Verifies LI_HOME environment variable is set before proceeding
    void LoadKernelLibrary(std::string_view folder_name, std::string_view context_name, std::string_view so_file_name);

private:
    ProfilerEngine()  = default;
    ~ProfilerEngine() = default;

    std::filesystem::path ck_root_path_;
    std::string           device_name_;

    std::filesystem::path           cache_file_path_;
    std::shared_ptr<ProfileCacheDB> profile_cache_ptr_;

    std::map<GemmOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_gemm_kernel_instance_map_;
    std::map<NormOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_norm_kernel_instance_map_;
    std::map<EmbeddingOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_embedding_kernel_instance_map_;
    std::map<FmhaOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_fmha_fwd_kernel_instance_map_;

    dylib kernel_lib_;
};

}  // namespace flashck