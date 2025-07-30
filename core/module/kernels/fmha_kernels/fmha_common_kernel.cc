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
FmhaCommonKernel::GenFmhaCommonKernelProfiler(const std::string&                               model_name,
                                              const std::unordered_map<std::string, std::any>& kernel_func_map,
                                              const std::string&                               create_args_source,
                                              const std::string&                               args_parser_source,
                                              const std::string&                               args_decl_source,
                                              const std::string&                               func_signature_source,
                                              const std::string&                               tensor_decl_source,
                                              const std::string&                               tensor_generate_source,
                                              const std::string&                               prepare_args_source,
                                              const std::string&                               func_call_source,
                                              const std::string&                               make_args_source,
                                              const std::string&                               fmha_flag,
                                              const std::string&                               folder_name)
{
    auto kernel_name = std::any_cast<std::string>(kernel_func_map.at("op_name"));
    auto op_mode     = std::any_cast<FmhaOperationMode>(kernel_func_map.at("op_mode"));
    auto dtype       = std::any_cast<DataType>(kernel_func_map.at("dtype"));

    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    std::filesystem::path prefix_path =
        std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / model_name / "profiler" / kernel_name;
    if (!std::filesystem::exists(prefix_path)) {
        std::filesystem::create_directories(prefix_path);
    }

    // common header file
    jinja2::ValuesMap common_header_value_map{{"dtype_config_utils", g_fmha_dtype_config_utils_source},
                                              {"rotary_utils", g_fmha_rotary_utils_source},
                                              {"seq_utils", g_fmha_seq_utils_source},
                                              {"args_decl", args_decl_source}};
    std::string       common_header = TemplateLoadAndRender(g_fmha_utils_source, common_header_value_map);

    std::filesystem::path common_header_path = prefix_path / ("fmha_" + fmha_flag + "_common.h");
    std::ofstream         common_header_file(common_header_path.c_str());
    if (common_header_file.is_open()) {
        common_header_file << common_header;
        common_header_file.close();
    }
    else {
        LI_THROW(Unavailable("unable to open file {}", ToString(common_header_path)));
    }

    // kernel instance file
    for (const auto& [kernel_config_name, kernel_instance] : kernel_instance_map) {
        std::string config, config_name;

        if (fmha_flag == "fwd") {
            auto fmha_fwd_kernel_instance = std::static_pointer_cast<FmhaFwdOperation>(kernel_instance);
            config                        = fmha_fwd_kernel_instance->Emit();
            config_name                   = fmha_fwd_kernel_instance->GetConfigName();
        }
        else if (fmha_flag == "fwd_appendkv") {
            auto fmha_fwd_kernel_instance = std::static_pointer_cast<FmhaFwdAppendKVOperation>(kernel_instance);
            config                        = fmha_fwd_kernel_instance->Emit();
            config_name                   = fmha_fwd_kernel_instance->GetConfigName();
        }
        else if (fmha_flag == "fwd_splitkv") {
            auto fmha_fwd_kernel_instance = std::static_pointer_cast<FmhaFwdSplitKVOperation>(kernel_instance);
            config                        = fmha_fwd_kernel_instance->Emit();
            config_name                   = fmha_fwd_kernel_instance->GetConfigName();
        }
        else if (fmha_flag == "fwd_splitkv_combine") {
            auto fmha_fwd_kernel_instance = std::static_pointer_cast<FmhaFwdSplitKVCombineOperation>(kernel_instance);
            config                        = fmha_fwd_kernel_instance->Emit();
            config_name                   = fmha_fwd_kernel_instance->GetConfigName();
        }
        else {
            LI_THROW(Unavailable("not implemented for operation kind"));
        }

        jinja2::ValuesMap dtype_decl_value_map{{"DataType", TileDataTypeToString(dtype)}};
        std::string       dtype_decl = TemplateLoadAndRender(g_fmha_dtype_decl_source, dtype_decl_value_map);
        // VLOG(1) << "dtype_decl:" << dtype_decl;

        jinja2::ValuesMap instances_decl_value_map{
            {"kernel_name", "FmhaInstance"}, {"config_name", config_name}, {"config", config}};
        std::string instances_decl = TemplateLoadAndRender(g_fmha_instance_source, instances_decl_value_map);
        // VLOG(1) << "instances_decl:" << instances_decl;

        jinja2::ValuesMap exec_value_map{{"kernel_name", "FmhaInstance"},
                                         {"is_profile_kernel", "true"},
                                         {"prepare_args", prepare_args_source},
                                         {"make_args", make_args_source},
                                         {"is_execute", false},
                                         {"config_name", config_name}};
        std::string       exec_program = TemplateLoadAndRender(g_fmha_execute_source, exec_value_map);
        // VLOG(1) << "exec_program:" << exec_program;

        jinja2::ValuesMap func_value_map{{"dtype_decl", dtype_decl},
                                         {"instances_decl", instances_decl},
                                         {"func_signature", func_signature_source},
                                         {"execute_func", exec_program},
                                         {"fmha_flag", fmha_flag}};
        std::string       kernel_func = TemplateLoadAndRender(g_fmha_kernel_func, func_value_map);
        // VLOG(1) << "kernel_func:" << kernel_func;

        jinja2::ValuesMap tensor_decl_value_map{{"fmha_flag", fmha_flag},
                                                {"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                                {"decl_source", tensor_decl_source}};
        std::string       tensor_decl = TemplateLoadAndRender(g_fmha_tenosr_decl_source, tensor_decl_value_map);
        // VLOG(1) << "tensor_decl: " << tensor_decl;

        jinja2::ValuesMap profiler_value_map{{"create_args", create_args_source},
                                             {"kernel_func", kernel_func},
                                             {"args_parser", args_parser_source},
                                             {"tensor_generate", tensor_generate_source},
                                             {"tensor_decl", tensor_decl},
                                             {"func_call", func_call_source}};
        std::string       profiler_source = TemplateLoadAndRender(g_fmha_profiler_source, profiler_value_map);
        // VLOG(1) << "profiler_source: " << profiler_source;

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

std::string
FmhaCommonKernel::GenFmhaCommonKernelFunction(const std::string&                               func_name,
                                              const std::string&                               model_name,
                                              const std::unordered_map<std::string, std::any>& kernel_func_map,
                                              const std::string&                               args_decl_source,
                                              const std::string&                               func_signature_source,
                                              const std::string&                               prepare_args_source,
                                              const std::string&                               make_args_source,
                                              const std::string&                               fmha_flag,
                                              const std::string&                               folder_name)
{
    auto kernel_name = std::any_cast<std::string>(kernel_func_map.at("op_name"));
    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));
    auto exec_path = std::any_cast<std::map<std::string, std::shared_ptr<ExecItem>>>(kernel_func_map.at("exec_path"));
    auto op_mode   = std::any_cast<FmhaOperationMode>(kernel_func_map.at("op_mode"));
    auto bias_enum = std::any_cast<BiasEnum>(kernel_func_map.at("bias_enum"));
    auto bias_rank_info   = std::any_cast<int64_t>(kernel_func_map.at("bias_rank_info"));
    auto dtype            = std::any_cast<DataType>(kernel_func_map.at("dtype"));
    auto paged_block_size = std::any_cast<int64_t>(kernel_func_map.at("paged_block_size"));

    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / model_name;
    if (!std::filesystem::exists(prefix_path)) {
        std::filesystem::create_directories(prefix_path);
    }

    // common header file
    jinja2::ValuesMap common_header_value_map{{"dtype_config_utils", g_fmha_dtype_config_utils_source},
                                              {"args_decl", args_decl_source}};
    std::string       common_header = TemplateLoadAndRender(g_fmha_utils_source, common_header_value_map);

    std::filesystem::path common_header_path = prefix_path / ("fmha_" + fmha_flag + "_common.h");
    std::ofstream         common_header_file(common_header_path.c_str());
    if (common_header_file.is_open()) {
        common_header_file << common_header;
        common_header_file.close();
    }
    else {
        LI_THROW(Unavailable("unable to open file {}", ToString(common_header_path)));
    }

    // kernel instance file
    std::string                                             instance_decl;
    std::unordered_set<std::string>                         instance_def_flag;
    std::map<std::string, std::tuple<int64_t, std::string>> exec_instance_map;
    for (const auto& [key, value] : exec_path) {
        std::string instance_name = "f" + SHA1ToHexString(value->exec_cond_);
        std::string algo          = value->algo_;
        int64_t     split_k       = value->split_k_;

        std::string config, config_name;
        if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
            auto kernel_instance = kernel_instance_map.at(algo);
            if (fmha_flag == "fwd") {
                auto fmha_fwd_kernel_instance = std::static_pointer_cast<FmhaFwdOperation>(kernel_instance);
                config                        = fmha_fwd_kernel_instance->Emit();
                config_name                   = fmha_fwd_kernel_instance->GetConfigName();
            }
            else if (fmha_flag == "fwd_appendkv") {
                auto fmha_fwd_kernel_instance = std::static_pointer_cast<FmhaFwdAppendKVOperation>(kernel_instance);
                config                        = fmha_fwd_kernel_instance->Emit();
                config_name                   = fmha_fwd_kernel_instance->GetConfigName();
            }
            else if (fmha_flag == "fwd_splitkv") {
                auto fmha_fwd_kernel_instance = std::static_pointer_cast<FmhaFwdSplitKVOperation>(kernel_instance);
                config                        = fmha_fwd_kernel_instance->Emit();
                config_name                   = fmha_fwd_kernel_instance->GetConfigName();
            }
            else if (fmha_flag == "fwd_splitkv_combine") {
                auto fmha_fwd_kernel_instance =
                    std::static_pointer_cast<FmhaFwdSplitKVCombineOperation>(kernel_instance);
                config      = fmha_fwd_kernel_instance->Emit();
                config_name = fmha_fwd_kernel_instance->GetConfigName();
            }
            else {
                LI_THROW(Unavailable("not implemented for operation kind"));
            }
            instance_def_flag.insert(instance_name);
        }
        else {
            config      = "";
            config_name = "";
        }

        jinja2::ValuesMap instance_value_map{
            {"kernel_name", instance_name}, {"config_name", config_name}, {"config", config}};
        std::string instance = TemplateLoadAndRender(g_fmha_instance_source, instance_value_map);
        std::get<1>(exec_instance_map[value->exec_cond_]) = instance;
        std::get<0>(exec_instance_map[value->exec_cond_]) = split_k;

        instance_decl += instance;
    }

    std::string exec_paths;
    for (const auto& [exec_cond, profile_result] : exec_instance_map) {
        std::string instance_name = "f" + SHA1ToHexString(exec_cond);

        jinja2::ValuesMap prepare_args_value_map{{"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                                 {"is_execute", true},
                                                 {"num_splits", std::get<0>(profile_result)},
                                                 {"paged_block_size", paged_block_size},
                                                 {"bias_str", g_bias_enum_names_map.at(bias_enum)},
                                                 {"bias_rank_info", bias_rank_info}};
        std::string       prepare_args = TemplateLoadAndRender(prepare_args_source, prepare_args_value_map);

        jinja2::ValuesMap make_args_value_map{{"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                              {"kernel_name", instance_name}};
        std::string       make_args = TemplateLoadAndRender(make_args_source, make_args_value_map);

        jinja2::ValuesMap exec_value_map{{"kernel_name", instance_name},
                                         {"is_profile_kernel", "false"},
                                         {"prepare_args", prepare_args},
                                         {"make_args", make_args},
                                         {"is_execute", true}};
        std::string       exec_program = TemplateLoadAndRender(g_fmha_execute_source, exec_value_map);

        jinja2::ValuesMap exec_instance_value_map{{"cond", exec_cond}, {"program", exec_program}};
        std::string       exec_instance = TemplateLoadAndRender(g_fmha_exec_cond_source, exec_instance_value_map);

        exec_paths += exec_instance;
    }

    std::string macro_decl = TemplateLoadAndRender(g_fmha_macro_decl, {{}});

    jinja2::ValuesMap dtype_decl_value_map{{"DataType", TileDataTypeToString(dtype)}};
    std::string       dtype_decl = TemplateLoadAndRender(g_fmha_dtype_decl_source, dtype_decl_value_map);

    jinja2::ValuesMap func_signature_value_map{
        {"function_name", func_name}, {"is_execute", true}, {"c_flag", "extern \"C\""}};
    std::string func_signature = TemplateLoadAndRender(func_signature_source, func_signature_value_map);

    jinja2::ValuesMap kernel_func_value_map{{"macro_decl", macro_decl},
                                            {"dtype_decl", dtype_decl},
                                            {"instances_decl", instance_decl},
                                            {"func_signature", func_signature},
                                            {"execute_func", exec_paths},
                                            {"fmha_flag", fmha_flag}};

    return TemplateLoadAndRender(g_fmha_kernel_func, kernel_func_value_map);
}

}  // namespace lightinfer