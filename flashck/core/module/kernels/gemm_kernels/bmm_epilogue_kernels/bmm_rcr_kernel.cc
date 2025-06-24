#include "flashck/core/module/kernels/gemm_kernels/bmm_epilogue_kernels/bmm_rcr_kernel.h"

#include "flashck/core/module/kernels/gemm_kernels/layout.h"

namespace flashck {

std::map<std::string, std::shared_ptr<void>> BmmRCRKernel::Init(const OperationKind&   op_kind,
                                                                const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<GemmOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
BmmRCRKernel::GenKernelProfiler(const std::string&                               model_name,
                                const std::unordered_map<std::string, std::any>& kernel_func_map,
                                const std::string&                               folder_name)
{

    RCRLayout rcr_layout;
    return GenBmmCommonKernelProfiler(model_name, kernel_func_map, rcr_layout.GetBmmArgsParse());
}

std::string BmmRCRKernel::GenKernelFunction(const std::string&                               func_name,
                                            const std::string&                               model_name,
                                            const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenBmmKernelFunction(func_name, kernel_func_map);
}

void BmmRCRKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{

    auto bmm_args = std::get<GemmKernelArgs>(args);

    decltype(&Bmm) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    kernel_func(bmm_args.in_ptr_,
                bmm_args.weight_ptr_,
                bmm_args.out_ptr_,
                bmm_args.a_dim0_,
                bmm_args.a_dim1_,
                bmm_args.a_dim2_,
                bmm_args.b_dim0_,
                bmm_args.b_dim1_,
                bmm_args.b_dim2_,
                bmm_args.c_dim0_,
                bmm_args.c_dim1_,
                bmm_args.c_dim2_,
                bmm_args.stream_);
}

}  // namespace flashck