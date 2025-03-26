#pragma once

#include <vector>

#include "lightinfer/core/module/kernels/gemm_kernels/gemm_common_kernel.h"
#include "lightinfer/core/module/kernels/kernel_registry.h"

namespace lightinfer {

class SplitKGemmRCRKernel: public GemmCommonKernel {
public:
    SplitKGemmRCRKernel() = default;

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

}  // namespace lightinfer

LIGHTINFER_REGISTER_KERNEL(CK, split_k_gemm_rcr, lightinfer::SplitKGemmRCRKernel, RCR, _Float16, float, ushort);
