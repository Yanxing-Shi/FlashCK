#include "core/module/kernels/gemm_kernels/tile/gemm_common_kernel.h"

/// @brief Global configuration flags for tuning parameters
FC_DECLARE_string(FC_HOME_PATH);                 ///< Base path for storing generated files
FC_DECLARE_int32(FC_TUNING_NUM_COLD_ITERATION);  ///< Number of cold iterations for warmup
FC_DECLARE_int32(FC_TUNING_NUM_REPEATS);         ///< Number of repeated measurements
FC_DECLARE_bool(FC_TUNING_GPU_TIMER);            ///< Use GPU-based timing vs CPU timing
FC_DECLARE_bool(FC_TUNING_LOG);                  ///< Enable detailed logging during tuning
FC_DECLARE_bool(FC_TUNING_FLUSH_CACHE);          ///< Flush caches between measurements
FC_DECLARE_int32(FC_TUNING_ROTATING_COUNT);      ///< Rotation count for measurement stability

namespace flashck {

namespace tile{

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
FmhaCommonKernel::GenFmhaCommonKernelProfiler(const std::string&    model_name,
                                              const std::string&    kind_name,
                                              const instance_map_t& instance_map,
                                              const FmhaTuningTpl&      tuning_tpl,
                                              const std::string&    folder_name)
{
    auto gemm_instance_map = std::get<tile_gemm_codegen_map_t>(instance_map);

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    // kernel instance file
    for (const auto& [instance_name, instance] : kernel_instance_map) {
        std::string instance_code = instance.Emit();

        // Generate data type declarations
        jinja2::ValuesMap dtype_decl_value_map{
            {"a_dtype", DataTypeToTileString(instance.problem_.a_dtype_)},
            {"b_dtype", DataTypeToTileString(instance.problem_.b_dtype_)},
            {"c_dtype", DataTypeToTileString(instance.problem_.c_dtype_)},
            {"d0_dtype", DataTypeToTileString(instance.problem_.d0_dtype_)},
            {"d1_dtype", DataTypeToTileString(instance.problem_.d1_dtype_)}
        };
        std::string dtype_decl = TemplateLoadAndRender(g_tile_gemm_dtype_decl_tpl, dtype_decl_value_map);

        // Generate layout declarations
        jinja2::ValuesMap layout_decl_value_map{
            {"a_layout", LayoutToTileString(instance.problem_.a_layout_)},
            {"b_layout", LayoutToTileString(instance.problem_.b_layout_)},
            {"c_layout", LayoutToTileString(instance.problem_.c_layout_)},
            {"d0_layout", LayoutToTileString(instance.problem_.d0_layout_)},
            {"d1_layout", LayoutToTileString(instance.problem_.d1_layout_)}
        };
        std::string layout_decl = TemplateLoadAndRender(g_tile_gemm_layout_decl_tpl, layout_decl_value_map);

        // Generate instance declarations
        jinja2::ValuesMap instance_decl_value_map{{"instance_alias_name", "GemmInstance"},
                                                  {"instance_name", instance_name},
                                                  {"instance_code", instance_code}};
        std::string       instance_decl = TEMPLATE_CHECK(g_tile_gemm_instance_tpl, instance_decl_value_map);

        // Generate runtime execution configuration with profiling parameters
        jinja2::ValuesMap running_value_map{{"make_args", tuning_tpl.make_args_tpl_},
                                             {"instance_alias_name", "NormInstance"},
                                             {"is_running", false},
                                             {"instance_name", instance_name},
                                         {"log_level", FLAGS_FC_TUNING_LOG},
                                        {"cold_niters", FLAGS_FC_TUNING_NUM_COLD_ITERATION},
                                        {"nrepeat", FLAGS_FC_TUNING_NUM_REPEATS},
                                        {"is_gpu_timer", FLAGS_FC_TUNING_GPU_TIMER},
                                        {"flush_cache", FLAGS_FC_TUNING_FLUSH_CACHE},
                                        {"rotating_count", FLAGS_FC_TUNING_ROTATING_COUNT}};
        std::string       running_program = TEMPLATE_CHECK(g_fmha_running_tpl, running_value_map);

        jinja2::ValuesMap func_value_map{{"header", tuning_tpl.header_tpl_},
                                        {"dtype_decl", dtype_decl},
                                         {"layout_decl", layout_decl},
                                         {"instance_decl", instance_decl},
                                         {"func_signature", tuning_tpl.func_signature_tpl_},
                                         {"running_func", running_program}};
        std::string       kernel_func = TEMPLATE_CHECK(g_tile_gemm_kernel_func_tpl, func_value_map);

        jinja2::ValuesMap profiling_value_map{{"create_args", create_args_tpl},
                                             {"kernel_func", kernel_func},
                                             {"args_parser", tuning_tpl.args_parser_tpl_},
                                             {"tensor_decl", tuning_tpl.tensor_decl_tpl_},
                                             {"func_call", tuning_tpl.func_call_tpl_}};
        std::string       profiling_tpl = TEMPLATE_CHECK(g_fmha_profiling_tpl, profiling_value_map);

        // Setup output file paths and create directories
        std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name
                                            / "profiling" / kind_name / instance_name;
        FileManager::CreateDirectoryIfNotExists(prefix_path);

        std::filesystem::path src_path = prefix_path / (instance_name + ".cc");
        std::filesystem::path obj_path = prefix_path / instance_name;
        
        // Skip if object file already exists (avoid redundant compilation)
        if (std::filesystem::exists(obj_path)) {
            continue;
        }

        FileManager::WriteFile(src_path, profiling_tpl);
        file_tuples.push_back(std::make_tuple(src_path, obj_path));
    }

    return file_tuples;
}

std::string
FmhaCommonKernel::CommonCodeGenForRunning(const std::string&                      func_name,
                                        const std::string&                        model_name,
                                        const std::map<std::string, RunningItem>& running_infos,
                                        const instance_map_t&                     instance_map,
                                        const FmhaRunningTpl&                         running_tpl,
                                        const std::string&                        folder_name)
{
    // Process kernel instances and build conditional runtime code
    std::string                                             instance_decl;
    std::unordered_set<std::string>                         instance_def_flag;
    std::map<std::string, std::string> running_instance_map;
    auto                               gemm_instance_map = std::get<tile_gemm_codegen_map_t>(instance_map);

    // Generate instance declarations for each running condition
    for (const auto& [_, running_item] : running_infos) {
        std::string hash_running_cond = "f" + HashToHexString(running_item.running_cond_);
        std::string instance_name     = running_item.instance_name_;

        std::string instance_code;
        if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
            auto instance = kernel_instance_map.at(algo);
            instance_code = instance->Emit();
            instance_def_flag.insert(instance_name);
        }

        jinja2::ValuesMap instance_value_map{{"instance_alias_name", hash_running_cond},
                                             {"instance_name", instance_name},
                                             {"instance_code", instance_code}};
        std::string instance = TEMPLATE_CHECK(g_fmha_instance_tpl, instance_value_map);
        running_instance_map[running_item.running_cond_] = instance;

        instance_decl += instance;
    }

    std::string running_paths;
    for (const auto& [running_cond, _] : running_instance_map) {
        std::string instance_name = "f" + HashToHexString(running_cond);

        jinja2::ValuesMap running_value_map{{"instance_alias_name", instance_name},
                                         {"make_args", make_args},
                                         {"is_running", true}};
        std::string       running_program = TEMPLATE_CHECK(g_tile_gemm_running_tpl, running_value_map);

        jinja2::ValuesMap running_instance_value_map{{"cond", running_cond}, {"program", running_program}};
        std::string       running_instance = TEMPLATE_CHECK(g_fmha_exec_cond_tpl, running_instance_value_map);

        running_paths += running_instance;
    }

    std::string macro_decl = TEMPLATE_CHECK(g_fmha_macro_decl, {{}});

    // Extract common configuration from first instance (assumes all instances share these properties)
    auto           instance           = gemm_instance_map.begin()->second;

    // Generate data type declarations
        jinja2::ValuesMap dtype_decl_value_map{
            {"a_dtype", DataTypeToTileString(instance.problem_.a_dtype_)},
            {"b_dtype", DataTypeToTileString(instance.problem_.b_dtype_)},
            {"c_dtype", DataTypeToTileString(instance.problem_.c_dtype_)},
            {"d0_dtype", DataTypeToTileString(instance.problem_.d0_dtype_)},
            {"d1_dtype", DataTypeToTileString(instance.problem_.d1_dtype_)}
        };
        std::string dtype_decl = TemplateLoadAndRender(g_tile_gemm_dtype_decl_tpl, dtype_decl_value_map);

        // Generate layout declarations
        jinja2::ValuesMap layout_decl_value_map{
            {"a_layout", LayoutToTileString(instance.problem_.a_layout_)},
            {"b_layout", LayoutToTileString(instance.problem_.b_layout_)},
            {"c_layout", LayoutToTileString(instance.problem_.c_layout_)},
            {"d0_layout", LayoutToTileString(instance.problem_.d0_layout_)},
            {"d1_layout", LayoutToTileString(instance.problem_.d1_layout_)}
        };
        std::string layout_decl = TemplateLoadAndRender(g_tile_gemm_layout_decl_tpl, layout_decl_value_map);


    jinja2::ValuesMap kernel_func_value_map{{"macro_decl", macro_decl},
                                            {"dtype_decl", dtype_decl},
                                            {"layout_decl", layout_decl},
                                            {"c_flag", "extern \"C\""},
                                            {"instances_decl", instance_decl},
                                            {"func_signature", running_tpl.func_signature_tpl_},
                                            {"running_func", running_paths},
                                            {"is_running", true}};

    return TEMPLATE_CHECK(g_tile_gemm_kernel_func_tpl, kernel_func_value_map);
}

} // namespace tile

}  // namespace flashck