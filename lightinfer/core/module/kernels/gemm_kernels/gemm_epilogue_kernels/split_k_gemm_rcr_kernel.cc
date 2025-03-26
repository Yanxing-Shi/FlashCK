#include "lightinfer/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/split_k_gemm_rcr_kernel.h"

#include "lightinfer/core/profiler/library.h"

#include "lightinfer/core/module/kernels/gemm_kernels/layout.h"

#include "lightinfer/core/utils/timer.h"

namespace lightinfer {

std::map<std::string, std::shared_ptr<void>> SplitKGemmRCRKernel::Init(const OperationKind&   op_kind,
                                                                       const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<GemmOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
SplitKGemmRCRKernel::GenKernelProfiler(const std::string&                               model_name,
                                       const std::unordered_map<std::string, std::any>& kernel_func_map,
                                       const std::string&                               folder_name)
{
    RCRLayout rcr_layout;
    return GenGemmCommonKernelProfiler(model_name, kernel_func_map, rcr_layout.GetSplitKGemmArgsParse(), "split_k");
}

std::string SplitKGemmRCRKernel::GenKernelFunction(const std::string&                               func_name,
                                                   const std::string&                               model_name,
                                                   const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenGemmCommonKernelFunction(func_name, kernel_func_map, "split_k");
}

void SplitKGemmRCRKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    auto gemm_args = std::get<GemmKernelArgs>(args);

    VLOG(1) << gemm_args.GetDimInfo();

    decltype(&SplitKGemm) kernel_func = nullptr;

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