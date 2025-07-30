#pragma once

#include "lightinfer/core/module/kernels/gemm_kernels/gemm_common_kernel.h"
#include "lightinfer/core/module/kernels/kernel_registry.h"

/*
GEMM ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[m, n] = tanh(a[m, k] * b[n, k] + bias[n])
This is used for `torch.nn.functional.linear + tanh`
When used for `linear`, need to set A->Data, B->Weight, C->Bias
*/

namespace flashck {

class GemmMultiDKernel: public GemmCommonKernel {
public:
    GemmMultiDKernel() = default;

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

LIGHTINFER_REGISTER_KERNEL(CK, gemm_rcr_bias_tanh, flashck::GemmMultiDKernel, RCR, _Float16, float, ushort)