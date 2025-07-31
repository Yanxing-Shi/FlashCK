#include "core/module/kernels/fmha_kernels/fmha_common_kernel.h"

/// @brief Global configuration flags for tuning parameters
FC_DECLARE_string(FC_HOME_PATH);                 ///< Base path for storing generated files
FC_DECLARE_int32(FC_TUNING_NUM_COLD_ITERATION);  ///< Number of cold iterations for warmup
FC_DECLARE_int32(FC_TUNING_NUM_REPEATS);         ///< Number of repeated measurements
FC_DECLARE_bool(FC_TUNING_GPU_TIMER);            ///< Use GPU-based timing vs CPU timing
FC_DECLARE_bool(FC_TUNING_LOG);                  ///< Enable detailed logging during tuning
FC_DECLARE_bool(FC_TUNING_FLUSH_CACHE);          ///< Flush caches between measurements
FC_DECLARE_int32(FC_TUNING_ROTATING_COUNT);      ///< Rotation count for measurement stability

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
FmhaCommonKernel::GenFmhaCommonKernelProfiler(const std::string&    model_name,
                                              const std::string&    kind_name,
                                              const instance_map_t& instance_map,
                                              const FmhaTuningTpl&      tuning_tpl,
                                              const std::string&    folder_name)
{
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    // common header file
    jinja2::ValuesMap common_header_value_map{{"dtype_config_utils", g_fmha_dtype_config_utils_tpl},
                                              {"rotary_utils", g_fmha_rotary_utils_tpl},
                                              {"seq_utils", g_fmha_seq_utils_tpl},
                                              {"args_decl", args_decl_tpl}};
    std::string       common_header = TEMPLATE_CHECK(g_fmha_utils_tpl, common_header_value_map);

    // kernel instance file
    for (const auto& [instance_name, instance] : kernel_instance_map) {
        std::string instance_code = instance.Emit();

        // Generate data type declarations
        jinja2::ValuesMap dtype_decl_value_map{{"DataType", DataTypeToTileString(instance.problem_.dtype_)}};
        std::string       dtype_decl = TEMPLATE_CHECK(g_fmha_dtype_decl_tpl, dtype_decl_value_map); 

        // Generate instance declarations
        jinja2::ValuesMap instance_decl_value_map{{"instance_alias_name", "FmhaInstance"},
                                                  {"instance_name", instance_name},
                                                  {"instance_code", instance_code}};
        std::string       instance_decl = TEMPLATE_CHECK(g_fmha_instance_tpl, instance_decl_value_map);

        // Generate runtime execution configuration with profiling parameters
        jinja2::ValuesMap running_value_map{{"kind", kind_name},
                                             {"prepare_args", prepare_args_tpl},
                                             {"make_args", make_args_tpl},
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

        jinja2::ValuesMap func_value_map{{"dtype_decl", dtype_decl},
                                         {"instance_decl", instance_decl},
                                         {"func_signature", func_signature_tpl},
                                         {"running_func", running_program}};
        std::string       kernel_func = TEMPLATE_CHECK(g_fmha_kernel_func_tpl, func_value_map);

        jinja2::ValuesMap profiling_value_map{{"create_args", create_args_tpl},
                                             {"kernel_func", kernel_func},
                                             {"args_parser", tuning_tpl.args_parser_tpl_},
                                             {"tensor_decl", tuning_tpl.tensor_decl_tpl_},
                                             {"tensor_generate", tuning_tpl.tensor_generate_tpl_},
                                             {"func_call", tuning_tpl.func_call_tpl_}};
        std::string       profiling_tpl = TEMPLATE_CHECK(g_fmha_profiling_tpl, profiling_value_map);

        // Setup output file paths and create directories
        std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name
                                            / "profiling" / kind_name / instance_name;
        FileManager::CreateDirectoryIfNotExists(prefix_path);

        // Write common header file (shared across instances)
        std::filesystem::path common_header_path = prefix_path / "fmha_common.h";
        FileManager::WriteFile(common_header_path, common_header);

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
    // Setup output directory and common header
    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / model_name;
    FileManager::CreateDirectoryIfNotExists(prefix_path);

    // common header file
    jinja2::ValuesMap common_header_value_map{{"dtype_config_utils", g_fmha_dtype_config_utils_tpl},
                                              {"args_decl", args_decl_tpl}};
    std::string       common_header = TEMPLATE_CHECK(g_fmha_utils_tpl, common_header_value_map);

    std::filesystem::path common_header_path = prefix_path / "fmha_common.h";
    std::ofstream         common_header_file(common_header_path.c_str());
    FileManager::WriteFile(common_header_path, common_header);


    // Process kernel instances and build conditional runtime code
    std::string                                             instance_decl;
    std::unordered_set<std::string>                         instance_def_flag;
    std::map<std::string, std::string> running_instance_map;
    auto                               fmha_instance_map = std::get<fmha_codegen_map_t>(instance_map);

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

    // Extract common configuration from first instance (assumes all instances share these properties)
    auto           instance           = fmha_instance_map.begin()->second;
    auto           mode          = GetFmhaModeName(instance.mode_);
    DataType       dtype            = instance.dtype_;

    std::string running_paths;
    for (const auto& [running_cond, _] : running_instance_map) {
        std::string instance_name = "f" + HashToHexString(running_cond);

        jinja2::ValuesMap running_value_map{{"instance_alias_name", instance_name},
                                         {"prepare_args", running_tpl.prepare_args_tpl_},
                                         {"make_args", make_args},
                                         {"is_running", true}};
        std::string       running_program = TEMPLATE_CHECK(g_fmha_running_tpl, running_value_map);

        jinja2::ValuesMap running_instance_value_map{{"cond", running_cond}, {"program", running_program}};
        std::string       running_instance = TEMPLATE_CHECK(g_fmha_exec_cond_tpl, running_instance_value_map);

        running_paths += running_instance;
    }

    std::string macro_decl = TEMPLATE_CHECK(g_fmha_macro_decl, {{}});

    jinja2::ValuesMap dtype_decl_value_map{{"DataType", DataTypeToTileString(dtype)}};
    std::string       dtype_decl = TEMPLATE_CHECK(g_fmha_dtype_decl_tpl, dtype_decl_value_map);

    jinja2::ValuesMap kernel_func_value_map{{"macro_decl", running_tpl.macro_decl_tpl_},
                                            {"dtype_decl", running_tpl.dtype_decl_tpl_},
                                            {"c_flag", "extern \"C\""},
                                            {"instances_decl", instance_decl},
                                            {"func_signature", running_tpl.func_signature_tpl_},
                                            {"running_func", running_paths},
                                            {"is_running", true}};

    return TEMPLATE_CHECK(g_fmha_kernel_func, kernel_func_value_map);
}

}  // namespace lightinfer