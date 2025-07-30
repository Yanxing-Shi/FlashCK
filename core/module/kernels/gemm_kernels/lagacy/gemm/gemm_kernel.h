#pragma once

#include <vector>

#include "core/module/kernels/gemm_kernels/lagacy/gemm_common_kernel.h"
#include "core/module/kernels/kernel_registry.h"

namespace flashck {

class GemmKernel: public GemmCommonKernel {
public:
    GemmKernel() = default;

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

FC_REGISTER_KERNEL(LEGACY, gemm, flashck::GemmKernel, ALL_LAYOUT, FP16, FP32, BF16);