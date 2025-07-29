#pragma once

#include "core/module/kernels/kernel.h"
#include "core/module/kernels/kernel_registry.h"
#include "core/module/kernels/norm_kernels/norm_common_kernel.h"

namespace flashck {

/**
 * @brief LayerNorm kernel implementation
 *
 * Implements LayerNormalization operation with support for:
 * - Multiple data types (FP16, FP32, BF16)
 * - Optional bias addition
 * - Residual connections
 * - Quantization support
 */
class LayerNormKernel: public NormCommonKernel {
public:
    /// @brief Generate tuning code for LayerNorm kernel
    /// @param model_name Name of the model being tuned
    /// @param kind_name Kind/type identifier ("layer_norm")
    /// @param instance_map Map of kernel instances and configurations
    /// @param folder_name Output folder for generated code
    /// @return Vector of tuples containing source and object file paths
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const std::string&    model_name,
                     const std::string&    kind_name,
                     const instance_map_t& instance_map,
                     const std::string&    folder_name = "kernel_profile") override;

    /// @brief Generate runtime code for LayerNorm kernel
    /// @param func_name Function name for the generated kernel
    /// @param model_name Name of the model
    /// @param running_infos Runtime configuration information
    /// @param instance_map Map of kernel instances and configurations
    /// @param folder_name Output folder for generated code
    /// @return Generated source code as string
    std::string CodeGenForRunning(const std::string&                        func_name,
                                  const std::string&                        model_name,
                                  const std::map<std::string, RunningItem>& running_infos,
                                  const instance_map_t&                     instance_map,
                                  const std::string&                        folder_name = "kernel_profile") override;

    /// @brief Execute LayerNorm kernel with given arguments
    /// @param kernel_func_name Name of the kernel function to launch
    /// @param args Kernel arguments containing tensors and parameters
    void KernelLauncher(const std::string& kernel_func_name, const KernelArgs_t& args) override;
};

}  // namespace flashck

/// @brief Register LayerNorm kernel for TILE source with FP16, FP32, BF16 support
FC_REGISTER_KERNEL(TILE, layer_norm, flashck::LayerNormKernel, ALL_LAYOUT, FP16, FP32, BF16);
