#include "flashck/core/module/kernels/norm_kernels/layer_norm_kernel.h"

#include <unordered_set>

#include "flashck/core/utils/dylib_utils.h"
#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/jinjia2_utils.h"
#include "flashck/core/utils/log.h"

LI_DECLARE_string(LI_HOME_PATH);

namespace flashck {

std::map<std::string, std::shared_ptr<void>> LayerNormKernel::Init(const OperationKind&   op_kind,
                                                                   const TensorOperation& extra_kind)
{
    std::map<std::string, std::shared_ptr<void>> layernorm_kernels_map;

    auto target_kernel_instance_map = Target::Instance()->target_norm_kernel_instance_map_;
    VLOG(1) << "target_kernel_instance_map size: " << target_kernel_instance_map.size();
    auto extract_kernel_map = target_kernel_instance_map.at(std::get<NormOperationKind>(op_kind)).at(extra_kind);
    for (auto [kernel_config_name, kernel_instance] : extract_kernel_map) {
        VLOG(1) << "extract layer norm kernel: " << kernel_config_name;
        layernorm_kernels_map[kernel_config_name] = kernel_instance;
    }

    LOG(INFO) << "Init kernel, op_kind: " << g_norm_operation_kind_names_map.at(std::get<NormOperationKind>(op_kind))
              << ", extra_kind: " << g_tensor_operation_names.at(extra_kind)
              << ", kernel size: " << layernorm_kernels_map.size();

    return layernorm_kernels_map;
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
LayerNormKernel::GenKernelProfiler(const std::string&                               model_name,
                                   const std::unordered_map<std::string, std::any>& kernel_func_map,
                                   const std::string&                               folder_name)
{
    return GenCommonKernelProfiler(model_name,
                                   kernel_func_map,
                                   g_layer_norm_dtype_config_utils_source,
                                   g_layer_norm_dtype_decl_source,
                                   g_layer_norm_func_signature_source,
                                   g_layer_norm_make_args_source,
                                   g_layer_norm_tensor_decl_source,
                                   g_layer_norm_func_call_source,
                                   folder_name);
}

std::string LayerNormKernel::GenKernelFunction(const std::string&                               func_name,
                                               const std::string&                               model_name,
                                               const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenCommonKernelFunction(func_name,
                                   model_name,
                                   kernel_func_map,
                                   g_layer_norm_dtype_config_utils_source,
                                   g_layer_norm_dtype_decl_source,
                                   g_layer_norm_func_signature_source,
                                   g_layer_norm_make_args_source);
}

void LayerNormKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    auto kernel_args = std::get<NormKernelArgs>(args);

    decltype(&LayerNorm) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

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

    VLOG(1) << kernel_func_name << " kernel launch success";
}

}  // namespace flashck