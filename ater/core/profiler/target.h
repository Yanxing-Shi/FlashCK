#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "ater/core/profiler/emitters.h"
#include "ater/core/profiler/generator.h"
#include "ater/core/profiler/profile_cache.h"

namespace ater {

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
    void GenerateKernel(const GenOperationKind& op_kind, const GemmProblem& gemm_problem);

    /*----------------compile-----------------*/
    std::vector<std::string> GetCKPaths();
    std::filesystem::path    GetROCmPkgPath();
    std::string              BuildComipleOptions(bool is_profile = true);
    std::string
    CompileCmd(const std::string& target, const std::string& src, bool enable_execute = false, bool is_profile = true);

    /*-----------Profiler cache-------------*/
    std::filesystem::path GetProfileCachePath();
    void                  LoadProfileCache();
    int                   GetProfileCacheVersion(const std::string& op_kind);
    std::tuple<std::string, int, int>
         QueryProfileCache(const std::string&                                                     op_kind,
                           const std::unordered_map<std::string, std::variant<int, std::string>>& args);
    void InsertProfileCache(const std::string&                                                     op_kind,
                            const std::unordered_map<std::string, std::variant<int, std::string>>& args);

    std::map<OperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        target_kernel_instance_map_;

private:
    std::filesystem::path ck_root_path_;
    std::string           device_name_;

    std::filesystem::path           cache_file_path_;
    std::shared_ptr<ProfileCacheDB> profile_cache_ptr_;
};

}  // namespace ater