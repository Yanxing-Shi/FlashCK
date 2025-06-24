#pragma once

#include "flashck/core/module/kernels/gemm_kernels/gemm_common_kernel.h"
#include "flashck/core/module/kernels/kernel_registry.h"

/*
GEMM ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[m, n] = a[m, k] * b[n, k] + bias[n]
This is used for `torch.nn.functional.linear`
When used for `linear`, need to set A->Data, B->Weight, C->Bias
*/

namespace flashck {

class GemmRCRBiasKernel: public GemmCommonKernel {
public:
    GemmRCRBiasKernel() = default;

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

flashck_REGISTER_KERNEL(CK, gemm_rcr_bias, flashck::GemmRCRBiasKernel, RCR, _Float16, float, ushort);