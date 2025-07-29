#pragma once

#include "core/module/kernels/kernel.h"
#include "core/module/kernels/kernel_registry.h"

#include "core/module/kernels/norm_kernels/norm_common_jinja.h"

namespace flashck {

/**
 * @brief Common base class for normalization kernel implementations
 *
 * Provides shared code generation functionality for LayerNorm, RMSNorm,
 * and other normalization operations.
 */
class NormCommonKernel: public Kernel {
public:
    /// @brief Generate tuning code for normalization kernels
    /// @param model_name Name of the model being tuned
    /// @param kind_name Kind/type identifier of the normalization
    /// @param instance_map Map of kernel instances and configurations
    /// @param tuning_tpl Template configuration for tuning
    /// @param folder_name Output folder for generated code
    /// @return Vector of tuples containing source and object file paths
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CommonCodeGenForTuning(const std::string&    model_name,
                           const std::string&    kind_name,
                           const instance_map_t& instance_map,
                           const TuningTpl&      tuning_tpl,
                           const std::string&    folder_name = "kernel_profile");

    /// @brief Generate runtime code for normalization kernels
    /// @param func_name Function name for the generated kernel
    /// @param model_name Name of the model
    /// @param running_infos Runtime configuration information
    /// @param instance_map Map of kernel instances and configurations
    /// @param running_tpl Template configuration for runtime
    /// @param folder_name Output folder for generated code
    /// @return Generated source code as string
    std::string CommonCodeGenForRunning(const std::string&                        func_name,
                                        const std::string&                        model_name,
                                        const std::map<std::string, RunningItem>& running_infos,
                                        const instance_map_t&                     instance_map,
                                        const RunningTpl&                         running_tpl,
                                        const std::string&                        folder_name = "kernel_profile");
};
}  // namespace flashck
