#include "flashck/core/module/kernels/norm_kernels/rms_norm_kernel.h"

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
RMSNormKernel::GenKernelProfiler(const std::string&                               model_name,
                                 const std::unordered_map<std::string, std::any>& kernel_func_map,
                                 const std::string&                               folder_name)
{
    return GenCommonKernelProfiler(model_name,
                                   kernel_func_map,
                                   g_rms_norm_dtype_config_utils_source,
                                   g_rms_norm_dtype_decl_source,
                                   g_rms_norm_func_signature_source,
                                   g_rms_norm_make_args_source,
                                   g_rms_norm_tensor_decl_source,
                                   g_rms_norm_func_call_source,
                                   folder_name);
}

std::string RMSNormKernel::GenKernelFunction(const std::string&                               func_name,
                                             const std::string&                               model_name,
                                             const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenCommonKernelFunction(func_name,
                                   model_name,
                                   kernel_func_map,
                                   g_rms_norm_dtype_config_utils_source,
                                   g_rms_norm_dtype_decl_source,
                                   g_rms_norm_func_signature_source,
                                   g_rms_norm_make_args_source);
}

void RMSNormKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    auto kernel_args = std::get<NormKernelArgs>(args);

    decltype(&RMSNorm) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    kernel_func(kernel_args.x_ptr_,
                kernel_args.x_residual_ptr_,
                kernel_args.smooth_scale_ptr_,
                kernel_args.gamma_ptr_,
                kernel_args.y_ptr_,
                kernel_args.y_residual_ptr_,
                kernel_args.y_scale_ptr_,
                kernel_args.x_dim_0_,
                kernel_args.x_dim_1_,
                kernel_args.eps_,
                kernel_args.x_stride_,
                kernel_args.xr_stride_,
                kernel_args.y_stride_,
                kernel_args.yr_stride_,
                kernel_args.stream_);

    VLOG(1) << kernel_func_name << " kernel launch success";
}

}  // namespace flashck