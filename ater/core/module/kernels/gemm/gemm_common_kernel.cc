#include "ater/core/module/kernels/gemm/gemm_common_kernel.h"

#include <memory>
#include <unordered_set>

#include "ater/core/utils/flags.h"
#include "ater/core/utils/jinjia2_utils.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/string_utils.h"

#include "ater/core/profiler/target.h"

ATER_DECLARE_string(ATER_HOME_PATH);

namespace ater {

/*
Extract (operation name, operation instance) pair
 from all operation candidates.

 Parameters
 ----------
 op_kind : ck_lib.library.OperationKind
     Operation kind.
 extra_kind : ck_lib.library.[AnyKind]
     Used to as extra flag to distinguish kernels.
     E.g. bias_add_relu vs. add_relu_bias
 f_prop_op: function
     Used to filter operation.

 Returns
 -------
 Dict
     Extracted (operation name, operation instance) pair.
*/

std::map<std::string, std::shared_ptr<void>> GemmKernelCommon::ExtractConfig(const OperationKind&   op_kind,
                                                                             const TensorOperation& extra_kind)
{
    std::map<std::string, std::shared_ptr<void>> gemm_kernels_map;
    VLOG(1) << "op_kind: " << static_cast<int>(op_kind) << ", extra_kind: " << static_cast<int>(extra_kind);

    auto target_kernel_instance_map = Target::Instance()->target_kernel_instance_map_;
    auto extract_kernel_map         = target_kernel_instance_map[op_kind][extra_kind];
    for (auto [kernel_config_name, kernel_instance] : extract_kernel_map) {
        // To Do: filter kernel
        VLOG(1) << "extract kernel: " << kernel_config_name;
        gemm_kernels_map[kernel_config_name] = kernel_instance;
    }

    VLOG(1) << "extract kernel size: " << gemm_kernels_map.size();
    return gemm_kernels_map;
}

// // Exract name from the statement, e.g. 'model' for 'using model = xxx'.
// const std::string GemmKernelCommon::ExetractConfigName(const std::string& config)
// {
//     std::regex               reg("\s*using\s(.*?)\s=");
//     std::vector<std::string> decl = Split(config, "\n");
//     std::smatch              config_match;
//     if (std::regex_match(decl[0], config_match, reg)) {

//         std::ssub_match sub_match   = pieces_match[0];
//         std::string     config_name = sub_match.str();
//         return config_name;
//     }
//     else {
//         throw std::runtime_error("Invalid config");
//     }
// }

//
std::string GemmKernelCommon::GenDimCalculator(const std::shared_ptr<DimInfo>& dim_info, bool is_ptr)
{
    std::string prefix = is_ptr ? "*" : "";
    if (dim_info->source_ == TensorSource::Input) {
        if (dim_info->tensor_idx_ == 0) {
            prefix += "a_dim";
        }
        else if (dim_info->tensor_idx_ == 1) {
            prefix += "b_dim";
        }
    }
    else if (dim_info->source_ == TensorSource::Output && dim_info->tensor_idx_ == 0) {
        prefix += "c_dim";
    }

    std::vector<std::string> dim_names_vec;
    for (int idx : dim_info->dim_idx_) {
        dim_names_vec.emplace_back("(" + prefix + std::to_string(idx) + ")");
    }

    std::string dim_names = JoinToString(dim_names_vec, "*");
    return dim_names;
}

std::string
GemmKernelCommon::GenShapeEvalCode(const std::string&                                                  dtype,
                                   const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
                                   bool                                                                is_ptr)
{
    std::vector<std::string> shape_eval_vec;
    for (auto [name, dim_info_vec] : dim_info_map) {
        std::shared_ptr<DimInfo> dim_info = std::make_shared<DimInfo>();
        for (auto d : dim_info_vec) {
            if (d->placeholder_) {
                continue;
            }
            dim_info = d;
            break;
        }

        jinja2::ValuesMap shape_eval_value_map{{"dtype", dtype},
                                               {"indent", "    "},
                                               {"name", name},
                                               {"dim_calculator", GenDimCalculator(dim_info, is_ptr)}};

        shape_eval_vec.emplace_back(TemplateLoadAndRender(g_shape_eval_source, shape_eval_value_map));
    }

    return JoinToString(shape_eval_vec, "\n");
}

/*
Generates standalone executables for profiler.

Parameters
----------
func_attrs : Dict
    Operation attributes.
workdir : str
    Directory to store the generated outputs.
dim_info_dict: Dict[str, DimInfo]
    Generated from gemm._extract_dims().
    Used to store mapping between dim_names to input / output tensor dims.
args_parse: str
    Profiler input argument parser.
gemm_flag : str
    Flag telling which backend should be generated. options are
'','bias','bias_relu','bias_sigmoid','bias_add_relu'. extra_code : str Extra code for self-defined operators. ndims
: int Number of dims for each parameter, 2 for gemm, 3 for bmm extra_shape_template: jinja2.Template Shape
evaluation template. problem_args_template: jinja2.Template Problem args template for profiler.
extra_header_template: jinja2.Template
    Extra header template as we have different headers for gemm and bmm.
tensor_decl_template: jinja2.Template
    Tensor declaration template.
*/

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> GemmKernelCommon::GenCommonKernelProfiler(
    const std::string&                                                  kernel_name,
    const std::string&                                                  model_name,
    const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
    const int                                                           num_sources,
    const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
    const std::string&                                                  arg_parse,
    const std::string&                                                  permute_shape,
    const std::string&                                                  extra_code,
    const std::string&                                                  gemm_flag,
    const int                                                           ndims,
    const std::string&                                                  extra_shape_template,
    const std::string&                                                  problem_args_template,
    const std::string&                                                  extra_header_template,
    const std::string&                                                  tensor_decl_template,
    const std::string&                                                  input_addr_calculator,
    const std::string&                                                  output_addr_calculator,
    const std::string&                                                  folder_name)
{
    // shape function
    std::string kernel_func_shape = GenShapeEvalCode("ck::index_t", dim_info_map, true);

    std::vector<std::string> adims;
    std::vector<std::string> bdims;
    std::vector<std::string> cdims;
    std::vector<std::string> pdims;

    adims.reserve(ndims);
    bdims.reserve(ndims);
    cdims.reserve(ndims);
    pdims.reserve(ndims);

    for (int i = 0; i < ndims; i++) {
        adims.push_back(std::string("&a_dim") + std::to_string(i));
        bdims.push_back(std::string("&b_dim") + std::to_string(i));
        cdims.push_back(std::string("&c_dim") + std::to_string(i));
    }

    if (!permute_shape.empty()) {
        for (int i = 0; i < permute_shape.size(); i++) {
            pdims.emplace_back("p_dim" + std::to_string(i));
        }
    }

    jinja2::ValuesMap extra_shape_value_map{{"indent", "    "}};
    std::string       extra_shape_func = TemplateLoadAndRender(extra_shape_template, extra_shape_value_map);

    bool has_d0_flag = num_sources >= 1 ? true : false;
    bool has_d1_flag = num_sources >= 2 ? true : false;

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    for (auto& [kernel_config_name, kernel_instance] : kernel_instance_map) {
        auto        gemm_kernel_instance = std::static_pointer_cast<GemmOperation>(kernel_instance);
        std::string config               = gemm_kernel_instance->Emit();
        std::string config_name          = gemm_kernel_instance->GetConfigName();

        jinja2::ValuesMap instance_value_map{
            {"name", "DeviceGemmInstance"}, {"config_name", config_name}, {"config", config}};
        std::string instance = TemplateLoadAndRender(g_instance_source, instance_value_map);

        jinja2::ValuesMap problem_args_value_map{
            {"indent", "    "}, {"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}, {"has_d1", has_d1_flag}};
        std::string problem_args = TemplateLoadAndRender(problem_args_template, problem_args_value_map);

        jinja2::ValuesMap exec_value_map{{"indent", "    "},
                                         {"instance", "DeviceGemmInstance"},
                                         {"problem_args", problem_args},
                                         {"is_profiler", true}};
        std::string       exec_program = TemplateLoadAndRender(g_exec_source, exec_value_map);

        jinja2::ValuesMap extra_header_value_map{{"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}};
        std::string       extra_header = TemplateLoadAndRender(extra_header_template, extra_header_value_map);

        jinja2::ValuesMap src_value_map{{"instances", instance},
                                        {"function_name", "gemm"},
                                        {"ndims", static_cast<int64_t>(ndims)},
                                        {"pdims", static_cast<int64_t>(pdims.size())},
                                        {"has_d0", has_d0_flag},
                                        {"has_d1", has_d1_flag},
                                        {"shape_func", kernel_func_shape},
                                        {"extra_shape", extra_shape_func},
                                        {"input_addr_calculator", input_addr_calculator},
                                        {"ouput_addr_calculator", output_addr_calculator},
                                        {"exec_paths", exec_program},
                                        {"extra_code", extra_code},
                                        {"gemm_flag", gemm_flag},
                                        {"extra_header", extra_header}};
        std::string       op_func = TemplateLoadAndRender(g_src_source, src_value_map);

        std::string structs_def = TemplateLoadAndRender(g_structs_source, {{}});

        jinja2::ValuesMap tensor_decl_value_map{
            {"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}, {"has_d1", has_d1_flag}};
        std::string tensor_decl = TemplateLoadAndRender(tensor_decl_template, tensor_decl_value_map);

        std::string       do_ptr_value = has_d0_flag ? "(void*)memory_pool->RequestHalfTensorByIdx(4)" : "";
        std::string       d1_ptr_value = has_d1_flag ? "(void*)memory_pool->RequestHalfTensorByIdx(5)" : "";
        auto              pdims_value  = pdims.empty() ? "" : jinja2::Reflect(pdims);
        jinja2::ValuesMap func_call_value_map{{"indent", "    "},
                                              {"func_name", "gemm"},
                                              {"in_ptr", "(void*)memory_pool->RequestHalfTensorByIdx(0)"},
                                              {"weight_ptr", "(void*)memory_pool->RequestHalfTensorByIdx(1)"},
                                              {"out_ptr", "(void*)memory_pool->RequestHalfTensorByIdx(2)"},
                                              {"bias_ptr", "(void*)memory_pool->RequestHalfTensorByIdx(3)"},
                                              {"d0_ptr", do_ptr_value},
                                              {"d1_ptr", d1_ptr_value},
                                              {"adims", jinja2::Reflect(adims)},
                                              {"bdims", jinja2::Reflect(bdims)},
                                              {"cdims", jinja2::Reflect(cdims)},
                                              {"pdims", pdims_value},
                                              {"gemm_flag", gemm_flag}};
        std::string       func_call = TemplateLoadAndRender(g_func_call_source, func_call_value_map);

        jinja2::ValuesMap profiler_value_map{{"structs_def", structs_def},
                                             {"op_func", op_func},
                                             {"args_parse", arg_parse},
                                             {"tensor_decl", tensor_decl},
                                             {"func_call", func_call},
                                             {"kernel_config_name", kernel_config_name}};
        std::string       code = TemplateLoadAndRender(g_profiler_source, profiler_value_map);

        std::filesystem::path prefix_path =
            std::filesystem::path(FLAGS_ATER_HOME_PATH) / folder_name / model_name / "profiler" / kernel_name;
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
            ATER_THROW(Unavailable("unable to open file {}", ToString(src_path)));
        }

        file_tuples.push_back(std::make_tuple(src_path, obj_path));
    }

    return file_tuples;
}

std::string GemmKernelCommon::GenCommonKernelFunction(
    const std::string&                                                  func_name,
    const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
    const int                                                           num_sources,
    const std::map<std::string, std::shared_ptr<ExecItem>>&             exec_path,
    const std::string                                                   permute_shape,
    const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
    const std::string&                                                  gemm_flag,
    const std::string&                                                  extra_code,
    const int                                                           ndims,
    const std::string&                                                  exec_cond_template,
    const std::string&                                                  extra_shape_template,
    const std::string&                                                  problem_args_template,
    const std::string&                                                  extra_header_template,
    const std::string&                                                  input_addr_calculator,
    const std::string&                                                  output_addr_calculator)
{
    std::map<std::string, std::string> instances_map;
    std::string                        instance_decl = "";
    std::unordered_set<std::string>    instance_def_flag;

    bool has_d0_flag = num_sources >= 1 ? true : false;
    bool has_d1_flag = num_sources >= 2 ? true : false;

    for (const auto& [_, value] : exec_path) {
        std::string instance_name = "f" + SHA1ToHexString(value->exec_cond_);
        std::string algo          = value->algo_;
        std::string config;
        std::string config_name;

        if (instance_def_flag.find(algo) == instance_def_flag.end()) {
            auto kernel_instance      = kernel_instance_map.at(algo);
            auto gemm_kernel_instance = std::static_pointer_cast<GemmOperation>(kernel_instance);
            config                    = gemm_kernel_instance->Emit();
            config_name               = gemm_kernel_instance->GetConfigName();
            instance_def_flag.insert(algo);
        }
        else {
            config      = "";
            config_name = "";
        }

        jinja2::ValuesMap instance_value_map{{"name", instance_name}, {"config_name", config_name}, {"config", config}};
        std::string       instance       = TemplateLoadAndRender(g_instance_source, instance_value_map);
        instances_map[value->exec_cond_] = instance;
        instance_decl += instance;
    }

    // jinja2::ValuesMap extra_shape_value_map{{"indent", "    "}};

    // std::string extra_shape_func = TemplateLoadAndRender(extra_shape_template, extra_shape_value_map);
    // std::string shape_eval_func  = GenShapeEvalCode("ck::index_t", dim_info_map, false);

    std::string exec_paths;
    // for (const auto& [key, value] : instances_map) {
    //     std::string instance_name = "f" + SHA1ToHexString(key);

    //     jinja2::ValuesMap problem_args_value_map{
    //         {"indent", "    "}, {"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}, {"has_d1", has_d1_flag}};
    //     std::string problem_args = TemplateLoadAndRender(problem_args_template, problem_args_value_map);

    //     jinja2::ValuesMap exec_value_map{
    //         {"indent", "    "}, {"instance", instance_name}, {"problem_args", problem_args}, {"is_profiler", false}};
    //     std::string exec_program = TemplateLoadAndRender(g_exec_source, exec_value_map);

    //     jinja2::ValuesMap exec_instance_value_map{{"indent", "    "}, {"cond", key}, {"program", exec_program}};
    //     std::string       exec_instance = TemplateLoadAndRender(exec_cond_template, exec_instance_value_map);

    //     exec_paths += exec_instance;
    // }

    for (const auto& [key, _] : instances_map) {
        std::string       instance_name = "f" + SHA1ToHexString(key);
        jinja2::ValuesMap exec_value_map{{"instance", instance_name}};

        std::string       exec_program = TemplateLoadAndRender(g_function_exec_source, exec_value_map);
        jinja2::ValuesMap exec_instance_value_map{{"indent", "    "}, {"cond", key}, {"program", exec_program}};

        std::string exec_instance = TemplateLoadAndRender(exec_cond_template, exec_instance_value_map);

        exec_paths += exec_instance;
    }

    jinja2::ValuesMap extra_header_value_map{{"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}};
    std::string       extra_header = TemplateLoadAndRender(extra_header_template, extra_header_value_map);

    int64_t p_dims = 0;
    if (!permute_shape.empty()) {
        p_dims = static_cast<int64_t>(permute_shape.size());
    }

    jinja2::ValuesMap src_value_map{{"instances", instance_decl},
                                    {"function_name", func_name},
                                    {"ndims", static_cast<int64_t>(ndims)},
                                    {"pdims", p_dims},
                                    {"has_d0", has_d0_flag},
                                    {"has_d1", has_d1_flag},
                                    {"input_addr_calculator", input_addr_calculator},
                                    {"ouput_addr_calculator", output_addr_calculator},
                                    {"exec_paths", exec_paths},
                                    {"extra_code", extra_code},
                                    {"gemm_flag", gemm_flag},
                                    {"extra_header", extra_header}};

    return TemplateLoadAndRender(g_function_source, src_value_map);
}
}  // namespace ater