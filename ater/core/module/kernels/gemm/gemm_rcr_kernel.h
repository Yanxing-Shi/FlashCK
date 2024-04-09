#pragma once

#include <vector>

#include "ater/core/module/kernels/gemm/gemm_common_kernel.h"
#include "ater/core/module/kernels/kernel_registry.h"

namespace ater {

class GemmRCRKernel: public GemmKernelCommon {
public:
    GemmRCRKernel() {}

    std::map<std::string, std::shared_ptr<void>> Init() override;

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenKernelProfiler(const std::string&                                                  kernel_type,
                      const std::string&                                                  model_name,
                      const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
                      const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map) override;

    std::string
    GenKernelFunction(const std::string&                                                  func_name,
                      const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
                      const std::map<std::string, std::shared_ptr<ExecItem>>&             exec_path,
                      const std::string                                                   permute_shape,
                      const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map) override;

    // void GenFunctionDecl() override {}

    // void GenFunctionCall() override {}

    bool FunctionFilter() override;
};

}  // namespace ater

ATER_REGISTER_KERNEL(CK, gemm_rcr, ater::GemmRCRKernel, RCR, _Float16, float);
