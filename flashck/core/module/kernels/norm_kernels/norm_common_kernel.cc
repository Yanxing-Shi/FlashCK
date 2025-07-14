#include "flashck/core/module/kernels/norm_kernels/layer_norm_kernel.h"

FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
NormCommonKernel::CommonCodeGenForTuning(const std::string&                                  model_name,
                                         const std::string&                                  kind_name,
                                         const std::map<std::string, std::unique_ptr<void>>& instance_map,
                                         const TuningTpl&                                    tuning_tpl,
                                         const std::string&                                  folder_name)
{

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    std::filesystem::path prefix_path =
        std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name / "profiling" / kind_name;
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

        jinja2::ValuesMap running_value_map{{"make_args", make_args},
                                            {"instance_alias_name", "NormInstance"},
                                            {"is_profiling", true},
                                            {"is_running", false},
                                            {"instance_name", instance_name}};
        std::string       running_program = TemplateLoadAndRender(g_norm_running_tpl, running_value_map);

        jinja2::ValuesMap func_signature_value_map{{"function_name", "Norm"}};
        std::string func_signature = TemplateLoadAndRender(tuning_tpl.func_signature_tpl_, func_signature_value_map);

        jinja2::ValuesMap func_value_map{{"dtype_decl", dtype_decl},
                                         {"instance_decl", instance_decl},
                                         {"func_signature", func_signature},
                                         {"execute_func", running_program},
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

std::string
NormCommonKernel::CommonCodeGenForRunning(const std::string&                                  func_name,
                                          const std::string&                                  model_name,
                                          const std::map<std::string, RunningItem>&           running_infos,
                                          const std::map<std::string, std::unique_ptr<void>>& kernel_instance_map,
                                          const RunningTpl&                                   running_tpl,
                                          const std::string&                                  folder_name)
{

    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name;
    FileManager::CreateDirectories(prefix_path);

    std::string common_header = TemplateLoadAndRender(running_tpl.dtype_config_tpl_, {{}});

    std::filesystem::path common_header_path = prefix_path / "norm_common.h";
    FileManager::WriteFile(common_header_path, common_header);

    std::string                        instance_decl;
    std::set<std::string>              instance_def_flag;
    std::map<std::string, std::string> running_instance_map;
    for (const auto& [_, running_item] : running_infos) {
        std::string hash_running_cond = "f" + HashToHexString(running_item.running_cond_);
        std::string instance_name     = running_item.instance_name_;

        std::string instance_code;
        if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
            auto* norm_kernel_instance = static_cast<NormCodeGen*>(kernel_instance_map.at(instance_name).get());
            instance_code              = norm_kernel_instance->Emit();
            instance_def_flag.insert(instance_name);
        }

        jinja2::ValuesMap instance_value_map{{"instance_alias_name", hash_running_cond},
                                             {"instance_name", instance_name},
                                             {"instance_code", instance_code}};
        std::string       instance = TemplateLoadAndRender(g_norm_instance_tpl, instance_value_map);
        running_instance_map[running_item.running_cond_] = instance;
        instance_decl += instance;
    }

    auto*          norm_kernel_instance = static_cast<NormCodeGen*>(kernel_instance_map.begin()->second.get());
    DataType       x_dtype              = norm_kernel_instance->x_dtype_;
    DataType       y_dtype              = norm_kernel_instance->y_dtype_;
    DataType       smooth_scale_dtype   = norm_kernel_instance->smooth_scale_dtype_;
    DataType       y_scale_dtype        = norm_kernel_instance->y_scale_dtype_;
    NormBiasEnum   is_add_bias          = norm_kernel_instance->is_add_bias_;
    FusedAddEnum   fused_add            = norm_kernel_instance->fused_add_;
    FusedQuantEnum fused_quant          = norm_kernel_instance->fused_quant_;

    std::string running_paths;
    for (const auto& [running_cond, _] : running_instance_map) {
        std::string instance_name = "f" + HashToHexString(running_cond);

        jinja2::ValuesMap make_args_value_map{{"is_add_bias", GetNormBiasName(is_add_bias)},
                                              {"fused_add", GetFusedAddName(fused_add)},
                                              {"fused_quant", GetFusedQuantName(fused_quant)}};
        std::string       make_args = TemplateLoadAndRender(running_tpl.make_args_tpl_, make_args_value_map);

        jinja2::ValuesMap running_value_map{{"make_args", make_args},
                                            {"instance_alias_name", instance_name},
                                            {"is_profiling", "false"},
                                            {"is_running", true}};
        std::string       running_program = TemplateLoadAndRender(g_norm_running_tpl, running_value_map);

        jinja2::ValuesMap running_instance_value_map{{"cond", running_cond}, {"program", running_program}};
        std::string       running_instance = TemplateLoadAndRender(g_norm_running_cond_tpl, running_instance_value_map);

        running_paths += running_instance;
    }

    std::string macro_decl = TemplateLoadAndRender(g_norm_macro_decl, {{}});

    jinja2::ValuesMap dtype_decl_value_map{{"x_dtype", DataTypeToTileString(x_dtype)},
                                           {"y_dtype", DataTypeToTileString(y_dtype)},
                                           {"smooth_scale_dtype", DataTypeToTileString(smooth_scale_dtype)},
                                           {"y_scale_dtype", DataTypeToTileString(y_scale_dtype)}};
    std::string       dtype_decl = TemplateLoadAndRender(running_tpl.dtype_decl_tpl_, dtype_decl_value_map);

    jinja2::ValuesMap func_signature_value_map{{"function_name", func_name}};
    std::string       func_signature = TemplateLoadAndRender(running_tpl.func_signature_tpl_, func_signature_value_map);

    jinja2::ValuesMap kernel_func_value_map{{"macro_decl", macro_decl},
                                            {"dtype_decl", dtype_decl},
                                            {"c_flag", "extern \"C\""},
                                            {"instance_decl", instance_decl},
                                            {"func_signature", func_signature},
                                            {"execute_func", running_paths},
                                            {"is_running", true}};

    return TemplateLoadAndRender(g_norm_kernel_func_tpl, kernel_func_value_map);
}

}  // namespace flashck