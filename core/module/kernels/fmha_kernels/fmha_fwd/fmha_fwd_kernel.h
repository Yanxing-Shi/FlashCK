#pragma once

#include "lightinfer/core/module/kernels/fmha_kernels/fmha_common_kernel.h"

#include "core/module/kernels/kernel_registry.h"
#include "core/module/kernels/kernel.h"
#include "core/module/kernels/fmha_kernels/fmha_common_kernel.h"

namespace flashck {
class FmhaFwdKernel: public FmhaCommonKernel {
public:
    FmhaFwdKernel()  = default;
    ~FmhaFwdKernel() = default;

    std::map<std::string, std::shared_ptr<void>> Init(const OperationKind&   op_kind,
                                                      const TensorOperation& extra_kind) override;

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenKernelProfiler(const std::string&                               model_name,
                      const std::unordered_map<std::string, std::any>& kernel_func_map,
                      const std::string&                               folder_name = "kernel_profile") override;

    std::string GenKernelFunction(const std::string&                               func_name,
                                  const std::string&                               model_name,
                                  const std::unordered_map<std::string, std::any>& kernel_func_map) override;

    void KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args) override;
};

}  // namespace flashck

LIGHTINFER_REGISTER_KERNEL(CK_TILE, fmha_fwd, flashck::FmhaFwdKernel, ALL_LAYOUT, _Float16, ushort);