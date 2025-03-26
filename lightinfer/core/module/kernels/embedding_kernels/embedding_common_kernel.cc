#include "lightinfer/core/module/kernels/embedding_kernels/embedding_common_kernel.h"

#include <unordered_set>

#include "lightinfer/core/utils/dylib_utils.h"
#include "lightinfer/core/utils/file_utils.h"
#include "lightinfer/core/utils/flags.h"
#include "lightinfer/core/utils/jinjia2_utils.h"
#include "lightinfer/core/utils/log.h"

#include "lightinfer/core/profiler/target.h"

LI_DECLARE_string(LI_HOME_PATH);

namespace lightinfer {

std::map<std::string, std::shared_ptr<void>> EmbeddingCommonKernel::ExtractConfig(const EmbeddingOperationKind& op_kind,
                                                                                  const TensorOperation& extra_kind)
{
    std::map<std::string, std::shared_ptr<void>> embedding_kernels_map;
    VLOG(1) << "op_kind: " << static_cast<int>(op_kind) << ", extra_kind: " << static_cast<int>(extra_kind);

    auto target_kernel_instance_map = Target::Instance()->target_embedding_kernel_instance_map_;
    auto extract_kernel_map         = target_kernel_instance_map[op_kind][extra_kind];
    for (auto [kernel_config_name, kernel_instance] : extract_kernel_map) {
        VLOG(1) << "extract embedding kernel: " << kernel_config_name;
        embedding_kernels_map[kernel_config_name] = kernel_instance;
    }

    VLOG(1) << "extract kernel size: " << embedding_kernels_map.size();

    return embedding_kernels_map;
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
EmbeddingCommonKernel::GenCommonKernelProfiler(const std::string&                               model_name,
                                               const std::unordered_map<std::string, std::any>& kernel_func_map,
                                               const std::string&                               embedding_flag,
                                               const std::string&                               folder_name)
{
    auto kernel_name = std::any_cast<std::string>(kernel_func_map.at("op_name"));
    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    for (const auto& [kernel_config_name, kernel_instance] : kernel_instance_map) {
        auto        embedding_kernel_instance = std::static_pointer_cast<EmbeddingOperation>(kernel_instance);
        std::string config                    = embedding_kernel_instance->Emit();
        std::string config_name               = embedding_kernel_instance->GetConfigName();

        std::string extra_header = TemplateLoadAndRender(g_extra_header_source, {{}});

        std::string profiler_header = TemplateLoadAndRender(g_profiler_header_source, {{}});

        jinja2::ValuesMap dtype_decl_value_map{
            {"emb_dtype", DataTypeToString(embedding_kernel_instance->emb_dtype_)},
            {"index_dtype", DataTypeToString(embedding_kernel_instance->index_dtype_)},
            {"gamma_dtype", DataTypeToString(embedding_kernel_instance->gamma_dtype_)},
            {"beta_dtype", DataTypeToString(embedding_kernel_instance->beta_dtype_)},
            {"acc_dtype", DataTypeToString(embedding_kernel_instance->acc_dtype_)},
            {"out_dtype", DataTypeToString(embedding_kernel_instance->y_dtype_)},
            {"embedding_flag", embedding_flag}};
        std::string dtype_decl = TemplateLoadAndRender(g_dtype_decl_source, dtype_decl_value_map);

        jinja2::ValuesMap instance_value_map{
            {"name", "DeviceEmbeddingInstance"}, {"config_name", config_name}, {"config", config}};
        std::string instance = TemplateLoadAndRender(g_instance_source, instance_value_map);

        jinja2::ValuesMap exec_value_map{
            {"indent", "    "}, {"instance", "DeviceEmbeddingInstance"}, {"embedding_flag", embedding_flag}};
        std::string exec_path = TemplateLoadAndRender(g_exec_source, exec_value_map);

        jinja2::ValuesMap func_value_map{{"extra_header", extra_header},
                                         {"profiler_header", profiler_header},
                                         {"dtype_decl", dtype_decl},
                                         {"instances_decl", instance},
                                         {"func_name", kernel_name},
                                         {"exec_path", exec_path},
                                         {"embedding_flag", embedding_flag},
                                         {"is_execute", false}};
        std::string       op_func = TemplateLoadAndRender(g_func_source, func_value_map);

        std::string structs_def = TemplateLoadAndRender(g_structs_source, {{}});

        jinja2::ValuesMap tensor_decl_value_map{{"embedding_flag", embedding_flag}};
        std::string       tensor_decl = TemplateLoadAndRender(g_tensor_decl_source, tensor_decl_value_map);

        jinja2::ValuesMap func_call_value_map{
            {"indent", "    "}, {"func_name", kernel_name}, {"embedding_flag", embedding_flag}};
        std::string func_call = TemplateLoadAndRender(g_func_call_source, func_call_value_map);

        jinja2::ValuesMap profiler_value_map{{"structs_def", structs_def},
                                             {"op_func", op_func},
                                             {"tensor_decl", tensor_decl},
                                             {"func_call", func_call},
                                             {"kernel_config_name", kernel_config_name},
                                             {"embedding_flag", embedding_flag}};
        std::string       code = TemplateLoadAndRender(g_profiler_source, profiler_value_map);

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
            src_file << code;
            src_file.close();
        }
        else {
            LI_THROW(Unavailable("unable to open file {}", ToString(src_path)));
        }

        file_tuples.push_back(std::make_tuple(src_path, obj_path));
    }

    VLOG(1) << "file_tuples size: " << file_tuples.size();
    return file_tuples;
}

std::string
EmbeddingCommonKernel::GenCommonKernelFunction(const std::string&                               func_name,
                                               const std::unordered_map<std::string, std::any>& kernel_func_map,
                                               const std::string&                               embedding_flag)
{
    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));
    auto exec_path = std::any_cast<std::map<std::string, std::shared_ptr<ExecItem>>>(kernel_func_map.at("exec_path"));

    std::string                        instances_decl;
    std::unordered_set<std::string>    instance_def_flag;
    std::map<std::string, std::string> exec_instance_map;
    for (const auto& [key, value] : exec_path) {
        std::string instance_name = "f" + SHA1ToHexString(value->exec_cond_);
        std::string algo          = value->algo_;

        std::string config, config_name;
        if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
            auto kernel_instance      = kernel_instance_map.at(algo);
            auto norm_kernel_instance = std::static_pointer_cast<EmbeddingOperation>(kernel_instance);
            config                    = norm_kernel_instance->Emit();
            config_name               = norm_kernel_instance->GetConfigName();
            instance_def_flag.insert(instance_name);
        }
        else {
            config      = "";
            config_name = "";
        }

        jinja2::ValuesMap instance_value_map{{"name", instance_name}, {"config_name", config_name}, {"config", config}};
        std::string       instance           = TemplateLoadAndRender(g_instance_source, instance_value_map);
        exec_instance_map[value->exec_cond_] = instance;
        instances_decl += instance;
    }

    std::string exec_paths;
    for (const auto& [exec_cond, _] : exec_instance_map) {
        std::string instance_name = "f" + SHA1ToHexString(exec_cond);

        jinja2::ValuesMap exec_value_map{
            {"indent", "    "}, {"instance", instance_name}, {"embedding_flag", embedding_flag}};
        std::string exec_program = TemplateLoadAndRender(g_exec_source, exec_value_map);

        jinja2::ValuesMap exec_instance_value_map{{"indent", "    "}, {"cond", exec_cond}, {"program", exec_program}};
        std::string       exec_instance = TemplateLoadAndRender(g_exec_cond_source, exec_instance_value_map);

        exec_paths += exec_instance;
    }

    std::string macro_decl = TemplateLoadAndRender(g_macro_decl_source, {{}});

    auto tmp_embedding_kernel_instance =
        std::static_pointer_cast<EmbeddingOperation>(kernel_instance_map.begin()->second);
    jinja2::ValuesMap dtype_decl_value_map{
        {"emb_dtype", DataTypeToString(tmp_embedding_kernel_instance->emb_dtype_)},
        {"index_dtype", DataTypeToString(tmp_embedding_kernel_instance->index_dtype_)},
        {"gamma_dtype", DataTypeToString(tmp_embedding_kernel_instance->gamma_dtype_)},
        {"beta_dtype", DataTypeToString(tmp_embedding_kernel_instance->beta_dtype_)},
        {"acc_dtype", DataTypeToString(tmp_embedding_kernel_instance->acc_dtype_)},
        {"out_dtype", DataTypeToString(tmp_embedding_kernel_instance->y_dtype_)},
        {"embedding_flag", embedding_flag}};
    std::string dtype_decl = TemplateLoadAndRender(g_dtype_decl_source, dtype_decl_value_map);

    std::string extra_header = TemplateLoadAndRender(g_extra_header_source, {{}});

    jinja2::ValuesMap src_value_map{{"macro_decl", macro_decl},
                                    {"c_flag", "extern \"C\""},
                                    {"extra_header", extra_header},
                                    {"instances_decl", instances_decl},
                                    {"dtype_decl", dtype_decl},
                                    {"func_name", func_name},
                                    {"exec_path", exec_paths},
                                    {"embedding_flag", embedding_flag},
                                    {"is_execute", true}};

    return TemplateLoadAndRender(g_func_source, src_value_map);
}

}  // namespace lightinfer
