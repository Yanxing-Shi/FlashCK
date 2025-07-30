#include "core/module/kernels/norm_kernels/layer_norm_kernel.h"

/// @brief Global configuration flags for tuning parameters
FC_DECLARE_string(FC_HOME_PATH);                 ///< Base path for storing generated files
FC_DECLARE_int32(FC_TUNING_NUM_COLD_ITERATION);  ///< Number of cold iterations for warmup
FC_DECLARE_int32(FC_TUNING_NUM_REPEATS);         ///< Number of repeated measurements
FC_DECLARE_bool(FC_TUNING_GPU_TIMER);            ///< Use GPU-based timing vs CPU timing
FC_DECLARE_bool(FC_TUNING_LOG);                  ///< Enable detailed logging during tuning
FC_DECLARE_bool(FC_TUNING_FLUSH_CACHE);          ///< Flush caches between measurements
FC_DECLARE_int32(FC_TUNING_ROTATING_COUNT);      ///< Rotation count for measurement stability

namespace flashck {

/// @brief Generate tuning code for normalization kernels
std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
NormCommonKernel::CommonCodeGenForTuning(const std::string&    model_name,
                                         const std::string&    kind_name,
                                         const instance_map_t& instance_map,
                                         const TuningTpl&      tuning_tpl,
                                         const std::string&    folder_name)
{
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    // Load common header template
    std::string common_header = TemplateLoadAndRender(tuning_tpl.dtype_config_tpl_, {{}});

    auto norm_instance_map = std::get<norm_codegen_map_t>(instance_map);

    // Generate code for each kernel instance
    for (const auto& [instance_name, instance] : norm_instance_map) {
        std::string instance_code = instance.Emit();

        // Generate data type declarations
        jinja2::ValuesMap dtype_decl_value_map{
            {"x_dtype", DataTypeToTileString(instance.x_dtype_)},
            {"y_dtype", DataTypeToTileString(instance.y_dtype_)},
            {"smooth_scale_dtype", DataTypeToTileString(instance.smooth_scale_dtype_)},
            {"y_scale_dtype", DataTypeToTileString(instance.y_scale_dtype_)}};
        std::string dtype_decl = TemplateLoadAndRender(tuning_tpl.dtype_decl_tpl_, dtype_decl_value_map);

        std::string create_args = TemplateLoadAndRender(g_norm_create_args_tpl, {{}});

        // Generate instance declarations
        jinja2::ValuesMap instance_decl_value_map{{"instance_alias_name", "NormInstance"},
                                                  {"instance_name", instance_name},
                                                  {"instance_code", instance_code}};
        std::string       instance_decl = TemplateLoadAndRender(g_norm_instance_tpl, instance_decl_value_map);

        // Generate argument creation code with feature flags
        jinja2::ValuesMap make_args_value_map{{"is_add_bias", GetNormBiasName(instance.is_add_bias_)},
                                              {"fused_add", GetFusedAddName(instance.fused_add_)},
                                              {"fused_quant", GetFusedQuantName(instance.fused_quant_)}};
        std::string       make_args = TemplateLoadAndRender(tuning_tpl.make_args_tpl_, make_args_value_map);

        // Generate runtime execution configuration with profiling parameters
        jinja2::ValuesMap running_value_map{{"kind", kind_name},
                                            {"make_args", make_args},
                                            {"instance_alias_name", "NormInstance"},
                                            {"is_profiling", true},
                                            {"is_running", false},
                                            {"instance_name", instance_name},
                                            {"log_level", FLAGS_FC_TUNING_LOG},
                                            {"cold_niters", FLAGS_FC_TUNING_NUM_COLD_ITERATION},
                                            {"nrepeat", FLAGS_FC_TUNING_NUM_REPEATS},
                                            {"is_gpu_timer", FLAGS_FC_TUNING_GPU_TIMER},
                                            {"flush_cache", FLAGS_FC_TUNING_FLUSH_CACHE},
                                            {"rotating_count", FLAGS_FC_TUNING_ROTATING_COUNT}};
        std::string       running_program = TemplateLoadAndRender(g_norm_running_tpl, running_value_map);

        // Generate function signature and kernel function templates
        jinja2::ValuesMap func_signature_value_map{{"function_name", "Norm"}};
        std::string func_signature = TemplateLoadAndRender(tuning_tpl.func_signature_tpl_, func_signature_value_map);

        jinja2::ValuesMap func_value_map{{"kind", kind_name},
                                         {"dtype_decl", dtype_decl},
                                         {"instance_decl", instance_decl},
                                         {"func_signature", func_signature},
                                         {"running_func", running_program},
                                         {"is_running", false}};
        std::string       kernel_func = TemplateLoadAndRender(g_norm_kernel_func_tpl, func_value_map);

        // Generate function call template with conditional feature support
        jinja2::ValuesMap func_call_value_map{{"function_name", "Norm"},
                                              {"is_add_bias", GetNormBiasName(instance.is_add_bias_)},
                                              {"fused_add", GetFusedAddName(instance.fused_add_)},
                                              {"fused_quant", GetFusedQuantName(instance.fused_quant_)}};
        std::string       func_call = TemplateLoadAndRender(tuning_tpl.func_call_tpl_, func_call_value_map);

        // Generate tensor declarations for profiling setup
        jinja2::ValuesMap tensor_decl_value_map{{"is_add_bias", GetNormBiasName(instance.is_add_bias_)},
                                                {"fused_add", GetFusedAddName(instance.fused_add_)},
                                                {"fused_quant", GetFusedQuantName(instance.fused_quant_)}};
        std::string       tensor_decl = TemplateLoadAndRender(tuning_tpl.tensor_decl_tpl_, tensor_decl_value_map);

        // Assemble complete profiling template
        jinja2::ValuesMap profiling_value_map{{"create_args", create_args},
                                              {"tensor_decl", tensor_decl},
                                              {"kernel_func", kernel_func},
                                              {"func_call", func_call}};
        std::string       profiler_tpl = TemplateLoadAndRender(g_norm_profiling_tpl, profiling_value_map);

        // Setup output file paths and create directories
        std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name
                                            / "profiling" / kind_name / instance_name;
        FileManager::CreateDirectoryIfNotExists(prefix_path);

        // Write common header file (shared across instances)
        std::filesystem::path common_header_path = prefix_path / "norm_common.h";
        FileManager::WriteFile(common_header_path, common_header);

        // Generate source and object file paths
        std::filesystem::path src_path = prefix_path / (instance_name + ".cc");
        std::filesystem::path obj_path = prefix_path / instance_name;

        // Skip if object file already exists (avoid redundant compilation)
        if (FileManager::FileExists(obj_path)) {
            continue;
        }

        // Write generated source code and track for compilation
        FileManager::WriteFile(src_path, profiler_tpl);
        file_tuples.push_back(std::make_tuple(src_path, obj_path));
    }

    return file_tuples;
}

/// @brief Generate runtime code for normalization kernels
std::string NormCommonKernel::CommonCodeGenForRunning(const std::string&                        func_name,
                                                      const std::string&                        model_name,
                                                      const std::map<std::string, RunningItem>& running_infos,
                                                      const instance_map_t&                     instance_map,
                                                      const RunningTpl&                         running_tpl,
                                                      const std::string&                        folder_name)
{
    // Setup output directory and common header
    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name;
    FileManager::CreateDirectoryIfNotExists(prefix_path);

    std::string common_header = TemplateLoadAndRender(running_tpl.dtype_config_tpl_, {{}});

    std::filesystem::path common_header_path = prefix_path / "norm_common.h";
    FileManager::WriteFile(common_header_path, common_header);

    // Process kernel instances and build conditional runtime code
    std::string                        instance_decl;
    std::set<std::string>              instance_def_flag;  ///< Track defined instances to avoid duplicates
    std::map<std::string, std::string> running_instance_map;
    auto                               norm_instance_map = std::get<norm_codegen_map_t>(instance_map);

    // Generate instance declarations for each running condition
    for (const auto& [_, running_item] : running_infos) {
        std::string hash_running_cond = "f" + HashToHexString(running_item.running_cond_);
        std::string instance_name     = running_item.instance_name_;

        // Generate instance code only once per unique instance
        std::string instance_code;
        if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
            auto instance = norm_instance_map.at(instance_name);
            instance_code = instance.Emit();
            instance_def_flag.insert(instance_name);
        }

        jinja2::ValuesMap instance_value_map{{"instance_alias_name", hash_running_cond},
                                             {"instance_name", instance_name},
                                             {"instance_code", instance_code}};
        std::string       instance = TemplateLoadAndRender(g_norm_instance_tpl, instance_value_map);
        running_instance_map[running_item.running_cond_] = instance;
        instance_decl += instance;
    }

    // Extract common configuration from first instance (assumes all instances share these properties)
    auto           instance           = norm_instance_map.begin()->second;
    auto           kind_name          = GetNormKindName(instance.kind_);
    DataType       x_dtype            = instance.x_dtype_;
    DataType       y_dtype            = instance.y_dtype_;
    DataType       smooth_scale_dtype = instance.smooth_scale_dtype_;
    DataType       y_scale_dtype      = instance.y_scale_dtype_;
    NormBiasEnum   is_add_bias        = instance.is_add_bias_;
    FusedAddEnum   fused_add          = instance.fused_add_;
    FusedQuantEnum fused_quant        = instance.fused_quant_;

    // Generate conditional runtime execution paths
    std::string running_paths;
    for (const auto& [running_cond, _] : running_instance_map) {
        std::string instance_name = "f" + HashToHexString(running_cond);

        // Generate runtime arguments with feature flags
        jinja2::ValuesMap make_args_value_map{{"is_add_bias", GetNormBiasName(is_add_bias)},
                                              {"fused_add", GetFusedAddName(fused_add)},
                                              {"fused_quant", GetFusedQuantName(fused_quant)}};
        std::string       make_args = TemplateLoadAndRender(running_tpl.make_args_tpl_, make_args_value_map);

        // Configure runtime execution (non-profiling mode)
        jinja2::ValuesMap running_value_map{{"kind", kind_name},
                                            {"make_args", make_args},
                                            {"instance_alias_name", instance_name},
                                            {"is_profiling", "false"},
                                            {"is_running", true}};
        std::string       running_program = TemplateLoadAndRender(g_norm_running_tpl, running_value_map);

        // Create conditional execution block
        jinja2::ValuesMap running_instance_value_map{{"cond", running_cond}, {"program", running_program}};
        std::string       running_instance = TemplateLoadAndRender(g_norm_running_cond_tpl, running_instance_value_map);

        running_paths += running_instance;
    }

    // Generate final kernel function with all components
    std::string macro_decl = TemplateLoadAndRender(g_norm_macro_decl, {{}});

    jinja2::ValuesMap dtype_decl_value_map{{"x_dtype", DataTypeToTileString(x_dtype)},
                                           {"y_dtype", DataTypeToTileString(y_dtype)},
                                           {"smooth_scale_dtype", DataTypeToTileString(smooth_scale_dtype)},
                                           {"y_scale_dtype", DataTypeToTileString(y_scale_dtype)}};
    std::string       dtype_decl = TemplateLoadAndRender(running_tpl.dtype_decl_tpl_, dtype_decl_value_map);

    jinja2::ValuesMap func_signature_value_map{{"function_name", func_name}};
    std::string       func_signature = TemplateLoadAndRender(running_tpl.func_signature_tpl_, func_signature_value_map);

    // Assemble complete runtime kernel function
    jinja2::ValuesMap kernel_func_value_map{{"kind", kind_name},
                                            {"macro_decl", macro_decl},
                                            {"dtype_decl", dtype_decl},
                                            {"c_flag", "extern \"C\""},
                                            {"instance_decl", instance_decl},
                                            {"func_signature", func_signature},
                                            {"running_func", running_paths},
                                            {"is_running", true}};

    return TemplateLoadAndRender(g_norm_kernel_func_tpl, kernel_func_value_map);
}

}  // namespace flashck