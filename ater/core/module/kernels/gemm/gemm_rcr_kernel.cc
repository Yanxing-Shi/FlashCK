#include "ater/core/module/kernels/gemm/gemm_rcr_kernel.h"

#include "ater/core/profiler/library.h"

#include "ater/core/module/kernels/gemm/layout.h"

namespace ater {

std::map<std::string, std::shared_ptr<void>> GemmRCRKernel::Init()
{
    auto op_kind    = OperationKind::Gemm;
    auto extra_kind = TensorOperation::PassThrough;
    return ExtractConfig(op_kind, extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmRCRKernel::GenKernelProfiler(const std::string&                                  kernel_name,
                                 const std::string&                                  model_name,
                                 const std::map<std::string, std::shared_ptr<void>>& kernel_instance_map,
                                 const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map)
{
    RCRLayout rcr_layout;
    return GenCommonKernelProfiler(
        kernel_name, model_name, kernel_instance_map, 0, dim_info_map, rcr_layout.args_parse);
}

std::string
GemmRCRKernel::GenKernelFunction(const std::string&                                      func_name,
                                 const std::map<std::string, std::shared_ptr<void>>&     kernel_instance_map,
                                 const std::map<std::string, std::shared_ptr<ExecItem>>& exec_path,
                                 const std::string                                       permute_shape,
                                 const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map)
{
    return GenCommonKernelFunction(
        func_name, kernel_instance_map, 0, exec_path, permute_shape, dim_info_map, "", "", 2);
}

// void GemmRCRKernel::GenFunctionDecl()

// void GemmRCRKernel::GenFunctionCall()

bool GemmRCRKernel::FunctionFilter()
{
    return true;
}

}  // namespace ater