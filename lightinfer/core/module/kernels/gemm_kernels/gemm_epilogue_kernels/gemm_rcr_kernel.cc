#include "lightinfer/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_kernel.h"

#include "lightinfer/core/profiler/library.h"

#include "lightinfer/core/module/kernels/gemm_kernels/layout.h"

#include "lightinfer/core/utils/timer.h"

namespace lightinfer {

std::map<std::string, std::shared_ptr<void>> GemmRCRKernel::Init(const OperationKind&   op_kind,
                                                                 const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<GemmOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmRCRKernel::GenKernelProfiler(const std::string&                               model_name,
                                 const std::unordered_map<std::string, std::any>& kernel_func_map,
                                 const std::string&                               folder_name)
{
    RCRLayout rcr_layout;
    return GenGemmCommonKernelProfiler(model_name, kernel_func_map, rcr_layout.GetGemmArgsParse());
}

std::string GemmRCRKernel::GenKernelFunction(const std::string&                               func_name,
                                             const std::string&                               model_name,
                                             const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenGemmCommonKernelFunction(func_name, kernel_func_map, "", "", 2);
}

void GemmRCRKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    auto gemm_args = std::get<GemmKernelArgs>(args);

    decltype(&Gemm) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    kernel_func(gemm_args.in_ptr_,
                gemm_args.weight_ptr_,
                gemm_args.out_ptr_,
                gemm_args.a_dim0_,
                gemm_args.a_dim1_,
                gemm_args.b_dim0_,
                gemm_args.b_dim1_,
                gemm_args.c_dim0_,
                gemm_args.c_dim1_,
                gemm_args.stream_);
}

}  // namespace lightinfer