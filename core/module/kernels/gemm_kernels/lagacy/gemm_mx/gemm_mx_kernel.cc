#include "core/module/kernels/gemm_kernels/legacy/gemm/gemm_kernel.h"


namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmKernel::GenKernelProfiler(const std::string&                               model_name,
                                 const std::unordered_map<std::string, std::any>& kernel_func_map,
                                 const std::string&                               folder_name)
{
    return GenGemmCommonKernelProfiler(model_name, kernel_func_map);
}

std::string GemmKernel::GenKernelFunction(const std::string&                               func_name,
                                             const std::string&                               model_name,
                                             const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenGemmCommonKernelFunction(func_name, kernel_func_map, "", "", 2);
}

void GemmKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
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

}  // namespace flashck