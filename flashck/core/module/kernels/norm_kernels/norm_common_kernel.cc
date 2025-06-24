#include "flashck/core/module/kernels/norm_kernels/layer_norm_kernel.h"

#include <unordered_set>

#include "flashck/core/utils/dylib_utils.h"
#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/jinjia2_utils.h"
#include "flashck/core/utils/log.h"

LI_DECLARE_string(LI_HOME_PATH);

namespace flashck {

std::map<std::string, std::shared_ptr<void>> NormCommonKernel::Init(const OperationKind&   op_kind,
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
NormCommonKernel::GenCommonKernelProfiler(const std::string&                               model_name,
                                          const std::unordered_map<std::string, std::any>& kernel_func_map,
                                          const std::string&                               dtype_config_source,
                                          const std::string&                               dtype_decl_source,
                                          const std::string&                               func_signature_source,
                                          const std::string&                               make_args_source,
                                          const std::string&                               tensor_decl_source,
                                          const std::string&                               func_call_source,
                                          const std::string&                               folder_name)
{
    auto kernel_name = std::any_cast<std::string>(kernel_func_map.at("op_name"));
    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));
    auto epsilon = std::any_cast<float>(kernel_func_map.at("eps"));

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    std::filesystem::path prefix_path =
        std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / model_name / "profiler" / kernel_name;
    if (!std::filesystem::exists(prefix_path)) {
        std::filesystem::create_directories(prefix_path);
    }

    // common header file
    std::string common_header = TemplateLoadAndRender(dtype_config_source, {{}});

    std::filesystem::path common_header_path = prefix_path / "norm_common.h";
    std::ofstream         common_header_file(common_header_path.c_str());
    if (common_header_file.is_open()) {
        common_header_file << common_header;
        common_header_file.close();
    }
    else {
        LI_THROW(Unavailable("unable to open file {}", ToString(common_header_path)));
    }

    for (const auto& [kernel_config_name, kernel_instance] : kernel_instance_map) {
        auto        norm_kernel_instance = std::static_pointer_cast<NormOperation>(kernel_instance);
        std::string config               = norm_kernel_instance->Emit();
        std::string config_name          = norm_kernel_instance->GetConfigName();

        jinja2::ValuesMap dtype_decl_value_map{
            {"x_dtype", TileDataTypeToString(norm_kernel_instance->x_dtype_)},
            {"y_dtype", TileDataTypeToString(norm_kernel_instance->y_dtype_)},
            {"smooth_scale_dtype", TileDataTypeToString(norm_kernel_instance->smooth_scale_dtype_)},
            {"y_scale_dtype", TileDataTypeToString(norm_kernel_instance->y_scale_dtype_)}};
        std::string dtype_decl = TemplateLoadAndRender(dtype_decl_source, dtype_decl_value_map);

        std::string create_args = TemplateLoadAndRender(g_norm_create_args_source, {{}});

        jinja2::ValuesMap instances_decl_value_map{
            {"kernel_name", "NormInstance"}, {"config_name", config_name}, {"config", config}};
        std::string instances_decl = TemplateLoadAndRender(g_norm_instance_source, instances_decl_value_map);

        jinja2::ValuesMap make_args_value_map{
            {"is_add_bias", g_tile_layer_norm_operation_kind_names_map.at(norm_kernel_instance->is_add_bias_)},
            {"fused_add_str", g_fused_add_enum_str_map.at(norm_kernel_instance->fused_add_)},
            {"fused_quant_str", g_fused_quant_enum_str_map.at(norm_kernel_instance->fused_quant_)}};
        std::string make_args = TemplateLoadAndRender(make_args_source, make_args_value_map);

        jinja2::ValuesMap exec_value_map{{"make_args", make_args},
                                         {"kernel_name", "NormInstance"},
                                         {"epsilon", epsilon},
                                         {"is_profile_kernel", "true"},
                                         {"is_execute", false},
                                         {"config_name", config_name}};
        std::string       exec_program = TemplateLoadAndRender(g_norm_execute_source, exec_value_map);

        jinja2::ValuesMap func_signature_value_map{{"function_name", "Norm"}};
        std::string       func_signature = TemplateLoadAndRender(func_signature_source, func_signature_value_map);

        jinja2::ValuesMap func_value_map{{"dtype_decl", dtype_decl},
                                         {"instances_decl", instances_decl},
                                         {"func_signature", func_signature},
                                         {"execute_func", exec_program},
                                         {"is_execute", false}};
        std::string       kernel_func = TemplateLoadAndRender(g_norm_kernel_func, func_value_map);

        jinja2::ValuesMap func_call_value_map{
            {"function_name", "Norm"},
            {"is_add_bias", g_tile_layer_norm_operation_kind_names_map.at(norm_kernel_instance->is_add_bias_)},
            {"fused_add_str", g_fused_add_enum_str_map.at(norm_kernel_instance->fused_add_)},
            {"fused_quant_str", g_fused_quant_enum_str_map.at(norm_kernel_instance->fused_quant_)}};
        std::string func_call = TemplateLoadAndRender(func_call_source, func_call_value_map);

        jinja2::ValuesMap tensor_decl_value_map{
            {"is_add_bias", g_tile_layer_norm_operation_kind_names_map.at(norm_kernel_instance->is_add_bias_)},
            {"fused_add_str", g_fused_add_enum_str_map.at(norm_kernel_instance->fused_add_)},
            {"fused_quant_str", g_fused_quant_enum_str_map.at(norm_kernel_instance->fused_quant_)}};
        std::string tensor_decl = TemplateLoadAndRender(tensor_decl_source, tensor_decl_value_map);

        jinja2::ValuesMap profiler_value_map{{"create_args", create_args},
                                             {"tensor_decl", tensor_decl},
                                             {"kernel_func", kernel_func},
                                             {"func_call", func_call}};
        std::string       profiler_source = TemplateLoadAndRender(g_norm_profiler_source, profiler_value_map);

        std::filesystem::path prefix_path =
            std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / model_name / "profiler" / kernel_name;
        if (!std::filesystem::exists(prefix_path)) {
            std::filesystem::create_directories(prefix_path);
        }

        std::filesystem::path src_path = prefix_path / (kernel_config_name + ".cc");
        std::filesystem::path obj_path = prefix_path / kernel_config_name;
        if (std::filesystem::exists(obj_path)) {
            continue;
        }

        std::ofstream src_file(src_path.c_str());
        if (src_file.is_open()) {
            src_file << profiler_source;
            src_file.close();
        }
        else {
            LI_THROW(Unavailable("unable to open file {}", ToString(src_path)));
        }

        file_tuples.push_back(std::make_tuple(src_path, obj_path));
    }

    return file_tuples;
}

std::string NormCommonKernel::GenCommonKernelFunction(const std::string&                               func_name,
                                                      const std::string&                               model_name,
                                                      const std::unordered_map<std::string, std::any>& kernel_func_map,
                                                      const std::string& dtype_config_source,
                                                      const std::string& dtype_decl_source,
                                                      const std::string& func_signature_source,
                                                      const std::string& make_args_source)
{
    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));
    auto exec_path   = std::any_cast<std::map<std::string, std::shared_ptr<ExecItem>>>(kernel_func_map.at("exec_path"));
    auto epsilon     = std::any_cast<float>(kernel_func_map.at("eps"));
    auto fused_add   = std::any_cast<FusedAddEnum>(kernel_func_map.at("fused_add"));
    auto fused_quant = std::any_cast<FusedQuantEnum>(kernel_func_map.at("fused_quant"));
    auto is_add_bias = std::any_cast<NormBiasEnum>(kernel_func_map.at("is_add_bias"));
    auto x_dtype     = std::any_cast<DataType>(kernel_func_map.at("x_dtype"));
    auto y_dtype     = std::any_cast<DataType>(kernel_func_map.at("y_dtype"));
    auto smooth_scale_dtype = std::any_cast<DataType>(kernel_func_map.at("smooth_scale_dtype"));
    auto y_scale_dtype      = std::any_cast<DataType>(kernel_func_map.at("y_scale_dtype"));

    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_LI_HOME_PATH) / "kernel_profile" / model_name;
    if (!std::filesystem::exists(prefix_path)) {
        std::filesystem::create_directories(prefix_path);
    }

    // common header file
    std::string common_header = TemplateLoadAndRender(dtype_config_source, {{}});

    std::filesystem::path common_header_path = prefix_path / "norm_common.h";
    std::ofstream         common_header_file(common_header_path.c_str());
    if (common_header_file.is_open()) {
        common_header_file << common_header;
        common_header_file.close();
    }
    else {
        LI_THROW(Unavailable("unable to open file {}", ToString(common_header_path)));
    }

    std::string                        instance_decl;
    std::unordered_set<std::string>    instance_def_flag;
    std::map<std::string, std::string> exec_instance_map;
    for (const auto& [key, value] : exec_path) {
        std::string instance_name = "f" + SHA1ToHexString(value->exec_cond_);
        std::string algo          = value->algo_;

        std::string config, config_name;
        if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
            auto kernel_instance      = kernel_instance_map.at(algo);
            auto norm_kernel_instance = std::static_pointer_cast<NormOperation>(kernel_instance);
            config                    = norm_kernel_instance->Emit();
            config_name               = norm_kernel_instance->GetConfigName();
            instance_def_flag.insert(instance_name);
        }
        else {
            config      = "";
            config_name = "";
        }

        jinja2::ValuesMap instance_value_map{
            {"kernel_name", instance_name}, {"config_name", config_name}, {"config", config}};
        std::string instance                 = TemplateLoadAndRender(g_norm_instance_source, instance_value_map);
        exec_instance_map[value->exec_cond_] = instance;
        instance_decl += instance;
    }

    std::string exec_paths;
    for (const auto& [exec_cond, _] : exec_instance_map) {
        std::string instance_name = "f" + SHA1ToHexString(exec_cond);

        jinja2::ValuesMap make_args_value_map{
            {"is_add_bias", g_tile_layer_norm_operation_kind_names_map.at(is_add_bias)},
            {"fused_add_str", g_fused_add_enum_str_map.at(fused_add)},
            {"fused_quant_str", g_fused_quant_enum_str_map.at(fused_quant)}};
        std::string make_args = TemplateLoadAndRender(make_args_source, make_args_value_map);

        jinja2::ValuesMap exec_value_map{{"make_args", make_args},
                                         {"kernel_name", instance_name},
                                         {"epsilon", epsilon},
                                         {"is_profile_kernel", "false"},
                                         {"is_execute", true}};
        std::string       exec_program = TemplateLoadAndRender(g_norm_execute_source, exec_value_map);

        jinja2::ValuesMap exec_instance_value_map{{"cond", exec_cond}, {"program", exec_program}};
        std::string       exec_instance = TemplateLoadAndRender(g_norm_exec_cond_source, exec_instance_value_map);

        exec_paths += exec_instance;
    }

    std::string macro_decl = TemplateLoadAndRender(g_norm_macro_decl, {{}});

    jinja2::ValuesMap dtype_decl_value_map{{"x_dtype", TileDataTypeToString(x_dtype)},
                                           {"y_dtype", TileDataTypeToString(y_dtype)},
                                           {"smooth_scale_dtype", TileDataTypeToString(smooth_scale_dtype)},
                                           {"y_scale_dtype", TileDataTypeToString(y_scale_dtype)}};
    std::string       dtype_decl = TemplateLoadAndRender(dtype_decl_source, dtype_decl_value_map);

    jinja2::ValuesMap func_signature_value_map{{"function_name", func_name}};
    std::string       func_signature = TemplateLoadAndRender(func_signature_source, func_signature_value_map);

    jinja2::ValuesMap kernel_func_value_map{{"macro_decl", macro_decl},
                                            {"dtype_decl", dtype_decl},
                                            {"c_flag", "extern \"C\""},
                                            {"instances_decl", instance_decl},
                                            {"func_signature", func_signature},
                                            {"execute_func", exec_paths},
                                            {"is_execute", true}};

    return TemplateLoadAndRender(g_norm_kernel_func, kernel_func_value_map);
}

}  // namespace flashck