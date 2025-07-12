#include "flashck/core/module/kernels/norm_kernels/layer_norm_kernel.h"

FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
NormCommonKernel::CommonCodeGenForTuning(const std::string&                                  model_name,
                                         const Problem&                                      problem,
                                         const std::map<std::string, std::unique_ptr<void>>& instance_map,
                                         const TuningTpl&                                    tuning_tpl,
                                         const std::string&                                  folder_name)
{

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name
                                        / "profiling" / GetNormKindName(std::get<NormProblem>(problem).kind_);
    FileManager::CreateDirectories(prefix_path);

    std::string common_header = TemplateLoadAndRender(tuning_tpl.dtype_config_tpl_, {{}});

    std::filesystem::path common_header_path = prefix_path / "norm_common.h";
    FileManager::WriteFile(common_header_path, common_header);

    for (const auto& [instance_name, instance] : instance_map) {
        auto*       norm_kernel_instance = static_cast<NormCodeGen*>(instance.get());
        std::string instance_code        = norm_kernel_instance->Emit();

        jinja2::ValuesMap dtype_decl_value_map{
            {"x_dtype", DataTypeToTileString(norm_kernel_instance->x_dtype_)},
            {"y_dtype", DataTypeToTileString(norm_kernel_instance->y_dtype_)},
            {"smooth_scale_dtype", DataTypeToTileString(norm_kernel_instance->smooth_scale_dtype_)},
            {"y_scale_dtype", DataTypeToTileString(norm_kernel_instance->y_scale_dtype_)}};
        std::string dtype_decl = TemplateLoadAndRender(tuning_tpl.dtype_decl_tpl_, dtype_decl_value_map);

        std::string create_args = TemplateLoadAndRender(g_norm_create_args_tpl, {{}});

        jinja2::ValuesMap instance_decl_value_map{{"instance_alias_name", "NormInstance"},
                                                  {"instance_name", instance_name},
                                                  {"instance_code", instance_code}};
        std::string       instance_decl = TemplateLoadAndRender(g_norm_instance_tpl, instance_decl_value_map);

        jinja2::ValuesMap make_args_value_map{{"is_add_bias", GetNormBiasName(norm_kernel_instance->is_add_bias_)},
                                              {"fused_add", GetFusedAddName(norm_kernel_instance->fused_add_)},
                                              {"fused_quant", GetFusedQuantName(norm_kernel_instance->fused_quant_)}};
        std::string       make_args = TemplateLoadAndRender(tuning_tpl.make_args_tpl_, make_args_value_map);

        jinja2::ValuesMap exec_value_map{{"make_args", make_args},
                                         {"instance_alias_name", "NormInstance"},
                                         {"is_profiling", true},
                                         {"is_running", false},
                                         {"instance_name", instance_name}};
        std::string       exec_program = TemplateLoadAndRender(g_norm_exec_tpl, exec_value_map);

        jinja2::ValuesMap func_signature_value_map{{"function_name", "Norm"}};
        std::string func_signature = TemplateLoadAndRender(tuning_tpl.func_signature_tpl_, func_signature_value_map);

        jinja2::ValuesMap func_value_map{{"dtype_decl", dtype_decl},
                                         {"instance_decl", instance_decl},
                                         {"func_signature", func_signature},
                                         {"execute_func", exec_program},
                                         {"is_running", false}};
        std::string       kernel_func = TemplateLoadAndRender(g_norm_kernel_func_tpl, func_value_map);

        jinja2::ValuesMap func_call_value_map{{"function_name", "Norm"},
                                              {"is_add_bias", GetNormBiasName(norm_kernel_instance->is_add_bias_)},
                                              {"fused_add", GetFusedAddName(norm_kernel_instance->fused_add_)},
                                              {"fused_quant", GetFusedQuantName(norm_kernel_instance->fused_quant_)}};
        std::string       func_call = TemplateLoadAndRender(tuning_tpl.func_call_tpl_, func_call_value_map);

        jinja2::ValuesMap tensor_decl_value_map{{"is_add_bias", GetNormBiasName(norm_kernel_instance->is_add_bias_)},
                                                {"fused_add", GetFusedAddName(norm_kernel_instance->fused_add_)},
                                                {"fused_quant", GetFusedQuantName(norm_kernel_instance->fused_quant_)}};
        std::string       tensor_decl = TemplateLoadAndRender(tuning_tpl.tensor_decl_tpl_, tensor_decl_value_map);

        jinja2::ValuesMap profiling_value_map{{"create_args", create_args},
                                              {"tensor_decl", tensor_decl},
                                              {"kernel_func", kernel_func},
                                              {"func_call", func_call}};
        std::string       profiler_tpl = TemplateLoadAndRender(g_norm_profiling_tpl, profiling_value_map);

        std::filesystem::path prefix_path =
            std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name / "profiling" / instance_name;
        FileManager::CreateDirectories(prefix_path);

        std::filesystem::path src_path = prefix_path / (instance_name + ".cc");
        std::filesystem::path obj_path = prefix_path / instance_name;
        if (FileManager::FileExists(obj_path)) {
            continue;
        }

        FileManager::WriteFile(src_path, profiler_tpl);

        file_tuples.push_back(std::make_tuple(src_path, obj_path));
    }

    return file_tuples;
}

// std::string NormCommonKernel::GenCommonKernelFunction(const std::string&                               func_name,
//                                                       const std::string&                               model_name,
//                                                       const std::unordered_map<std::string, std::any>&
//                                                       kernel_func_map, const std::string& dtype_config_tpl, const
//                                                       std::string&                               dtype_decl_tpl,
//                                                       const std::string& func_signature_tpl,
//                                                       const std::string& make_args_tpl)
// {
//     // auto kernel_instance_map =
//     //     std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));
//     // auto exec_path   = std::any_cast<std::map<std::string,
//     // std::shared_ptr<ExecItem>>>(kernel_func_map.at("exec_path")); auto epsilon     =
//     // std::any_cast<float>(kernel_func_map.at("eps")); auto fused_add   =
//     // std::any_cast<FusedAddEnum>(kernel_func_map.at("fused_add")); auto fused_quant =
//     // std::any_cast<FusedQuantEnum>(kernel_func_map.at("fused_quant")); auto is_add_bias =
//     // std::any_cast<NormBiasEnum>(kernel_func_map.at("is_add_bias")); auto x_dtype     =
//     // std::any_cast<DataType>(kernel_func_map.at("x_dtype")); auto y_dtype     =
//     // std::any_cast<DataType>(kernel_func_map.at("y_dtype")); auto smooth_scale_dtype =
//     // std::any_cast<DataType>(kernel_func_map.at("smooth_scale_dtype")); auto y_scale_dtype      =
//     // std::any_cast<DataType>(kernel_func_map.at("y_scale_dtype"));

//     // std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / "kernel_profile" /
//     // model_name; if (!std::filesystem::exists(prefix_path)) {
//     //     std::filesystem::create_directories(prefix_path);
//     // }

//     // // common header file
//     // std::string common_header = TemplateLoadAndRender(dtype_config_tpl, {{}});

//     // Write common header using FileManager
//     // std::filesystem::path common_header_path = prefix_path / "norm_common.h";
//     // FileManager::WriteFile(common_header_path, common_header);

//     // std::string                        instance_decl;
//     // std::unordered_set<std::string>    instance_def_flag;
//     // std::map<std::string, std::string> exec_instance_map;
//     // for (const auto& [key, value] : exec_path) {
//     //     std::string instance_name = "f" + SHA1ToHexString(value->exec_cond_);
//     //     std::string algo          = value->algo_;

//     //     std::string config, instance_name;
//     //     if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
//     //         auto kernel_instance      = kernel_instance_map.at(algo);
//     //         auto norm_kernel_instance = std::static_pointer_cast<NormOperation>(kernel_instance);
//     //         config                    = norm_kernel_instance->Emit();
//     //         instance_name               = norm_kernel_instance->GetInstanceName();
//     //         instance_def_flag.insert(instance_name);
//     //     }
//     //     else {
//     //         config      = "";
//     //         instance_name = "";
//     //     }

//     //     jinja2::ValuesMap instance_value_map{
//     //         {"instance_alias_name", instance_name}, {"instance_name", instance_name}, {"instance_code", config}};
//     //     std::string instance                 = TemplateLoadAndRender(g_norm_instance_tpl, instance_value_map);
//     //     exec_instance_map[value->exec_cond_] = instance;
//     //     instance_decl += instance;
//     // }

//     // std::string exec_paths;
//     // for (const auto& [exec_cond, _] : exec_instance_map) {
//     //     std::string instance_name = "f" + SHA1ToHexString(exec_cond);

//     //     jinja2::ValuesMap make_args_value_map{
//     //         {"is_add_bias", g_tile_layer_norm_operation_kind_names_map.at(is_add_bias)},
//     //         {"fused_add", g_fused_add_enum_str_map.at(fused_add)},
//     //         {"fused_quant", g_fused_quant_enum_str_map.at(fused_quant)}};
//     //     std::string make_args = TemplateLoadAndRender(make_args_tpl, make_args_value_map);

//     //     jinja2::ValuesMap exec_value_map{{"make_args", make_args},
//     //                                      {"instance_alias_name", instance_name},
//     //                                      {"epsilon", epsilon},
//     //                                      {"is_profiling", "false"},
//     //                                      {"is_running", true}};
//     //     std::string       exec_program = TemplateLoadAndRender(g_norm_exec_tpl, exec_value_map);

//     //     jinja2::ValuesMap exec_instance_value_map{{"cond", exec_cond}, {"program", exec_program}};
//     //     std::string       exec_instance = TemplateLoadAndRender(g_norm_exec_cond_tpl, exec_instance_value_map);

//     //     exec_paths += exec_instance;
//     // }

//     // std::string macro_decl = TemplateLoadAndRender(g_norm_macro_decl, {{}});

//     // jinja2::ValuesMap dtype_decl_value_map{{"x_dtype", DataTypeToTileString(x_dtype)},
//     //                                        {"y_dtype", DataTypeToTileString(y_dtype)},
//     //                                        {"smooth_scale_dtype", DataTypeToTileString(smooth_scale_dtype)},
//     //                                        {"y_scale_dtype", DataTypeToTileString(y_scale_dtype)}};
//     // std::string       dtype_decl = TemplateLoadAndRender(dtype_decl_tpl, dtype_decl_value_map);

//     // jinja2::ValuesMap func_signature_value_map{{"function_name", func_name}};
//     // std::string       func_signature = TemplateLoadAndRender(func_signature_tpl, func_signature_value_map);

//     // jinja2::ValuesMap kernel_func_value_map{{"macro_decl", macro_decl},
//     //                                         {"dtype_decl", dtype_decl},
//     //                                         {"c_flag", "extern \"C\""},
//     //                                         {"instance_decl", instance_decl},
//     //                                         {"func_signature", func_signature},
//     //                                         {"execute_func", exec_paths},
//     //                                         {"is_running", true}};

//     // return TemplateLoadAndRender(g_norm_kernel_func_tpl, kernel_func_value_map);
// }

}  // namespace flashck