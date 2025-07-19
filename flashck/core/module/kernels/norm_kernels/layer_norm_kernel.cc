#include "flashck/core/module/kernels/norm_kernels/layer_norm_kernel.h"

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
LayerNormKernel::CodeGenForTuning(const std::string&    model_name,
                                  const std::string&    kind_name,
                                  const instance_map_t& instance_map,
                                  const std::string&    folder_name)
{
    // Static template configuration for LayerNorm tuning
    static const std::vector<std::string> templates = {g_layer_norm_dtype_config_utils_tpl,
                                                       g_layer_norm_dtype_decl_tpl,
                                                       g_layer_norm_func_signature_tpl,
                                                       g_layer_norm_make_args_tpl,
                                                       g_layer_norm_tensor_decl_tpl,
                                                       g_layer_norm_func_call_tpl};

    return CommonCodeGenForTuning(model_name, kind_name, instance_map, templates, folder_name);
}

std::string LayerNormKernel::CodeGenForRunning(const std::string&                        func_name,
                                               const std::string&                        model_name,
                                               const std::map<std::string, RunningItem>& running_infos,
                                               const instance_map_t&                     instance_map,
                                               const std::string&                        folder_name)
{
    // Static template configuration for LayerNorm runtime (subset of tuning templates)
    static const std::vector<std::string> templates = {g_layer_norm_dtype_config_utils_tpl,
                                                       g_layer_norm_dtype_decl_tpl,
                                                       g_layer_norm_func_signature_tpl,
                                                       g_layer_norm_make_args_tpl};

    return CommonCodeGenForRunning(func_name, model_name, running_infos, instance_map, templates, folder_name);
}

void LayerNormKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs_t& args)
{
    const auto& kernel_args = std::get<NormKernelArgs>(args);

    // Load kernel function symbol dynamically
    decltype(&LayerNorm) kernel_func = nullptr;
    LOAD_SYMBOL(kernel_func, kernel_func_name);

    // Execute kernel with all LayerNorm arguments
    kernel_func(kernel_args.x_ptr_,
                kernel_args.x_residual_ptr_,
                kernel_args.smooth_scale_ptr_,
                kernel_args.x_bias_ptr_,
                kernel_args.gamma_ptr_,
                kernel_args.beta_ptr_,
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

    VLOG(1) << kernel_func_name << " kernel launched successfully";
}

}  // namespace flashck