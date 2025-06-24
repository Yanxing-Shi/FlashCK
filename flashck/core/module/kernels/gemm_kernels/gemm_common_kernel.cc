#include "flashck/core/module/kernels/gemm_kernels/gemm_common_kernel.h"

#include <memory>
#include <unordered_set>

#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/jinjia2_utils.h"
#include "flashck/core/utils/log.h"
#include "flashck/core/utils/printf.h"
#include "flashck/core/utils/rocm_info.h"
#include "flashck/core/utils/string_utils.h"
#include "flashck/core/utils/timer.h"

#include "flashck/core/profiler/target.h"

LI_DECLARE_string(LI_HOME_PATH);

namespace flashck {

/*
Extract (operation name, operation instance) pair
 from all operation candidates.

 Parameters
 ----------
 op_kind : ck_lib.library.GemmOperationKind
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

std::map<std::string, std::shared_ptr<void>> GemmCommonKernel::ExtractConfig(const GemmOperationKind& op_kind,
                                                                             const TensorOperation&   extra_kind)
{
    std::map<std::string, std::shared_ptr<void>> gemm_kernels_map;
    VLOG(1) << "op_kind: " << static_cast<int>(op_kind) << ", extra_kind: " << static_cast<int>(extra_kind);

    auto target_kernel_instance_map = Target::Instance()->target_gemm_kernel_instance_map_;
    VLOG(1) << "target_kernel_instance_map size: " << target_kernel_instance_map.size();

    auto extract_kernel_map = target_kernel_instance_map[op_kind][extra_kind];
    for (auto [kernel_config_name, kernel_instance] : extract_kernel_map) {
        VLOG(1) << "extract gemm kernel: " << kernel_config_name;
        gemm_kernels_map[kernel_config_name] = kernel_instance;
    }

    VLOG(1) << "extract kernel size: " << gemm_kernels_map.size();
    return gemm_kernels_map;
}

// // Exract name from the statement, e.g. 'model' for 'using model = xxx'.
// const std::string GemmCommonKernel::ExetractConfigName(const std::string& config)
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
std::string GemmCommonKernel::GenDimCalculator(const std::shared_ptr<DimInfo>& dim_info, bool is_ptr)
{
    std::string prefix = is_ptr ? "*" : "";
    if (dim_info->source_ == TensorSource::kInput) {
        if (dim_info->tensor_idx_ == 0) {
            prefix += "a_dim";
        }
        else if (dim_info->tensor_idx_ == 1) {
            prefix += "b_dim";
        }
        else if (dim_info->tensor_idx_ == 2) {
            prefix += "b1_dim";
        }
    }
    else if (dim_info->source_ == TensorSource::kOutput && dim_info->tensor_idx_ == 0) {
        prefix += "c_dim";
    }

    std::vector<std::string> dim_names_vec;
    for (auto idx : dim_info->dim_idx_) {
        dim_names_vec.emplace_back("(" + prefix + std::to_string(idx) + ")");
    }

    std::string dim_names = JoinToString(dim_names_vec, "*");
    return dim_names;
}

std::string
GemmCommonKernel::GenShapeEvalCode(const std::string&                                                  dtype,
                                   const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
                                   bool                                                                is_ptr)
{
    std::vector<std::string> shape_eval_vec;
    for (const auto& [name, dim_info_vec] : dim_info_map) {
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

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmCommonKernel::GenGemmCommonKernelProfiler(const std::string&                               model_name,
                                              const std::unordered_map<std::string, std::any>& kernel_func_map,
                                              const std::string&                               arg_parse,
                                              const std::string&                               gemm_flag,
                                              const std::string&                               extra_code,
                                              const int                                        ndims,
                                              const std::string&                               extra_shape_template,
                                              const std::string&                               problem_args_template,
                                              const std::string&                               extra_header_template,
                                              const std::string&                               tensor_decl_template,
                                              const std::string&                               inverse_shape,
                                              const std::string&                               input_addr_calculator,
                                              const std::string&                               output_addr_calculator,
                                              const std::string&                               folder_name)
{
    auto kernel_name = std::any_cast<std::string>(kernel_func_map.at("op_name"));
    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));
    auto num_sources = std::any_cast<int>(kernel_func_map.at("num_sources"));
    auto dim_info_map =
        std::any_cast<std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>>(kernel_func_map.at("dim_info_map"));
    auto permute_shape = std::any_cast<Shape>(kernel_func_map.at("permute_shape"));

    // shape function
    std::string kernel_func_shape = GenShapeEvalCode("ck::index_t", dim_info_map, false);

    std::vector<std::string> a_dims;
    std::vector<std::string> b_dims;
    std::vector<std::string> b1_dims;
    std::vector<std::string> c_dims;
    std::vector<std::string> p_dims;

    a_dims.reserve(ndims);
    b_dims.reserve(ndims);
    b1_dims.reserve(ndims);
    c_dims.reserve(ndims);

    for (int i = 0; i < ndims; i++) {
        a_dims.push_back(std::string("a_dim") + std::to_string(i));
        b_dims.push_back(std::string("b_dim") + std::to_string(i));
        b1_dims.push_back(std::string("b1_dim") + std::to_string(i));
        c_dims.push_back(std::string("c_dim") + std::to_string(i));
    }

    if (permute_shape.GetNumDim()) {
        for (int i = 0; i < permute_shape.GetNumDim(); i++) {
            p_dims.emplace_back("p_dim" + std::to_string(i));
        }
    }

    jinja2::ValuesMap extra_shape_value_map{{"indent", "    "}};
    std::string       extra_shape_func = TemplateLoadAndRender(extra_shape_template, extra_shape_value_map);

    bool has_d0_flag = num_sources >= 1 ? true : false;
    bool has_d1_flag = num_sources >= 2 ? true : false;

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;

    for (const auto& [kernel_config_name, kernel_instance] : kernel_instance_map) {
        auto        gemm_kernel_instance = std::static_pointer_cast<GemmOperation>(kernel_instance);
        std::string config               = gemm_kernel_instance->Emit();
        std::string config_name          = gemm_kernel_instance->GetConfigName();

        jinja2::ValuesMap extra_header_value_map{{"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}};
        std::string       extra_header = TemplateLoadAndRender(extra_header_template, extra_header_value_map);

        std::string profiler_header = TemplateLoadAndRender(g_profiler_header_source, {{}});

        jinja2::ValuesMap dtype_decl_value_map{
            {"a_dtype", DataTypeToString(gemm_kernel_instance->a_tensor_desc_.element_)},
            {"b_dtype", DataTypeToString(gemm_kernel_instance->b_tensor_desc_.element_)},
            {"c_dtype", DataTypeToString(gemm_kernel_instance->c_tensor_desc_.element_)},
        };
        std::string dtype_decl = TemplateLoadAndRender(g_dtype_decl_source, dtype_decl_value_map);

        jinja2::ValuesMap layout_decl_value_map{
            {"a_layout", g_layout_tag.find(gemm_kernel_instance->a_tensor_desc_.layout_)->second},
            {"b_layout", g_layout_tag.find(gemm_kernel_instance->b_tensor_desc_.layout_)->second},
            {"c_layout", g_layout_tag.find(gemm_kernel_instance->c_tensor_desc_.layout_)->second}};
        std::string layout_decl = TemplateLoadAndRender(g_layout_decl_source, layout_decl_value_map);

        jinja2::ValuesMap instance_value_map{
            {"name", "DeviceGemmInstance"}, {"config_name", config_name}, {"config", config}};
        std::string instance = TemplateLoadAndRender(g_instance_source, instance_value_map);

        jinja2::ValuesMap problem_args_value_map{{"indent", "    "},
                                                 {"gemm_flag", gemm_flag},
                                                 {"has_d0", has_d0_flag},
                                                 {"has_d1", has_d1_flag},
                                                 {"is_execute", false}};
        std::string       problem_args = TemplateLoadAndRender(problem_args_template, problem_args_value_map);

        jinja2::ValuesMap exec_value_map{
            {"indent", "    "}, {"instance", "DeviceGemmInstance"}, {"problem_args", problem_args}};
        std::string exec_program = TemplateLoadAndRender(g_exec_source, exec_value_map);

        jinja2::ValuesMap src_value_map{{"instances", instance},
                                        {"function_name", "gemm"},
                                        {"ndims", static_cast<int64_t>(ndims)},
                                        {"p_dims", static_cast<int64_t>(p_dims.size())},
                                        {"has_d0", has_d0_flag},
                                        {"has_d1", has_d1_flag},
                                        {"shape_func", kernel_func_shape},
                                        {"extra_shape", extra_shape_func},
                                        {"inverse_shape", inverse_shape},
                                        {"input_addr_calculator", input_addr_calculator},
                                        {"ouput_addr_calculator", output_addr_calculator},
                                        {"exec_paths", exec_program},
                                        {"extra_code", extra_code},
                                        {"gemm_flag", gemm_flag},
                                        {"profiler_header", profiler_header},
                                        {"dtype_decl", dtype_decl},
                                        {"layout_decl", layout_decl},
                                        {"extra_header", extra_header},
                                        {"is_execute", false}};
        std::string       op_func = TemplateLoadAndRender(g_src_source, src_value_map);

        std::string structs_def = TemplateLoadAndRender(g_structs_source, {{}});

        jinja2::ValuesMap tensor_decl_value_map{
            {"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}, {"has_d1", has_d1_flag}};
        std::string tensor_decl = TemplateLoadAndRender(tensor_decl_template, tensor_decl_value_map);

        auto              p_dims_value = p_dims.empty() ? "" : jinja2::Reflect(p_dims);
        jinja2::ValuesMap func_call_value_map{{"indent", "    "},
                                              {"func_name", "gemm"},
                                              {"has_d0", has_d0_flag},
                                              {"has_d1", has_d1_flag},
                                              {"a_dims", jinja2::Reflect(a_dims)},
                                              {"b_dims", jinja2::Reflect(b_dims)},
                                              {"b1_dims", jinja2::Reflect(b1_dims)},
                                              {"c_dims", jinja2::Reflect(c_dims)},
                                              {"p_dims", p_dims_value},
                                              {"gemm_flag", gemm_flag},
                                              {"is_execute", false}};
        std::string       func_call = TemplateLoadAndRender(g_func_call_source, func_call_value_map);

        jinja2::ValuesMap profiler_value_map{{"structs_def", structs_def},
                                             {"op_func", op_func},
                                             {"args_parse", arg_parse},
                                             {"extra_shape", extra_shape_func},
                                             {"tensor_decl", tensor_decl},
                                             {"func_call", func_call},
                                             {"kernel_config_name", kernel_config_name}};
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

    return file_tuples;
}

std::string
GemmCommonKernel::GenGemmCommonKernelFunction(const std::string&                               func_name,
                                              const std::unordered_map<std::string, std::any>& kernel_func_map,
                                              const std::string&                               gemm_flag,
                                              const std::string&                               extra_code,
                                              const int                                        ndims,
                                              const std::string&                               extra_shape_template,
                                              const std::string&                               problem_args_template,
                                              const std::string&                               extra_header_template,
                                              const std::string&                               inverse_shape,
                                              const std::string&                               exec_cond_template)
{
    auto kernel_instance_map =
        std::any_cast<std::map<std::string, std::shared_ptr<void>>>(kernel_func_map.at("kernel_instance_map"));
    auto num_sources = std::any_cast<int>(kernel_func_map.at("num_sources"));
    auto exec_path   = std::any_cast<std::map<std::string, std::shared_ptr<ExecItem>>>(kernel_func_map.at("exec_path"));
    auto dim_info_map =
        std::any_cast<std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>>(kernel_func_map.at("dim_info_map"));
    auto permute_shape = std::any_cast<Shape>(kernel_func_map.at("permute_shape"));

    std::string shape_eval_func = GenShapeEvalCode("ck::index_t", dim_info_map, false);

    bool has_d0_flag = num_sources >= 1 ? true : false;
    bool has_d1_flag = num_sources >= 2 ? true : false;

    std::string                                             instance_decl;
    std::unordered_set<std::string>                         instance_def_flag;
    std::map<std::string, std::tuple<int64_t, std::string>> exec_instance_map;

    for (const auto& [key, value] : exec_path) {
        VLOG(1) << "exec_path: " << key << ", " << value->algo_;
        std::string instance_name = "f" + SHA1ToHexString(value->exec_cond_);
        std::string algo          = value->algo_;
        int64_t     split_k       = value->split_k_;

        std::string config, config_name;
        if (instance_def_flag.find(instance_name) == instance_def_flag.end()) {
            auto kernel_instance      = kernel_instance_map.at(algo);
            auto gemm_kernel_instance = std::static_pointer_cast<GemmOperation>(kernel_instance);
            config                    = gemm_kernel_instance->Emit();
            config_name               = gemm_kernel_instance->GetConfigName();
            instance_def_flag.insert(instance_name);
        }
        else {
            config      = "";
            config_name = "";
        }

        jinja2::ValuesMap instance_value_map{{"name", instance_name}, {"config_name", config_name}, {"config", config}};
        std::string       instance = TemplateLoadAndRender(g_instance_source, instance_value_map);
        std::get<1>(exec_instance_map[value->exec_cond_]) = instance;
        std::get<0>(exec_instance_map[value->exec_cond_]) = split_k;
        instance_decl += instance;
    }

    int64_t p_dims = permute_shape.GetNumDim() ? permute_shape.GetNumDim() : 0;

    std::string exec_paths;
    for (const auto& [exec_cond, profile_result] : exec_instance_map) {
        std::string instance_name = "f" + SHA1ToHexString(exec_cond);

        jinja2::ValuesMap problem_args_value_map{{"indent", "    "},
                                                 {"gemm_flag", gemm_flag},
                                                 {"has_d0", has_d0_flag},
                                                 {"has_d1", has_d1_flag},
                                                 {"is_execute", true},
                                                 {"split_k", std::get<0>(profile_result)}};
        std::string       problem_args = TemplateLoadAndRender(problem_args_template, problem_args_value_map);

        jinja2::ValuesMap exec_value_map{
            {"indent", "    "}, {"instance", instance_name}, {"problem_args", problem_args}};
        std::string exec_program = TemplateLoadAndRender(g_exec_source, exec_value_map);

        jinja2::ValuesMap exec_instance_value_map{{"indent", "    "}, {"cond", exec_cond}, {"program", exec_program}};
        std::string       exec_instance = TemplateLoadAndRender(exec_cond_template, exec_instance_value_map);

        exec_paths += exec_instance;
    }

    std::string macro_decl = TemplateLoadAndRender(g_macro_decl_source, {{}});

    jinja2::ValuesMap extra_header_value_map{{"gemm_flag", gemm_flag}, {"has_d0", has_d0_flag}};
    std::string       extra_header = TemplateLoadAndRender(extra_header_template, extra_header_value_map);

    auto tmp_gemm_kernel_instance = std::static_pointer_cast<GemmOperation>(kernel_instance_map.begin()->second);

    jinja2::ValuesMap dtype_decl_value_map{
        {"a_dtype", DataTypeToString(tmp_gemm_kernel_instance->a_tensor_desc_.element_)},
        {"b_dtype", DataTypeToString(tmp_gemm_kernel_instance->b_tensor_desc_.element_)},
        {"c_dtype", DataTypeToString(tmp_gemm_kernel_instance->c_tensor_desc_.element_)},
    };
    std::string dtype_decl = TemplateLoadAndRender(g_dtype_decl_source, dtype_decl_value_map);

    jinja2::ValuesMap layout_decl_value_map{
        {"a_layout", g_layout_tag.find(tmp_gemm_kernel_instance->a_tensor_desc_.layout_)->second},
        {"b_layout", g_layout_tag.find(tmp_gemm_kernel_instance->b_tensor_desc_.layout_)->second},
        {"c_layout", g_layout_tag.find(tmp_gemm_kernel_instance->c_tensor_desc_.layout_)->second}};
    std::string layout_decl = TemplateLoadAndRender(g_layout_decl_source, layout_decl_value_map);

    jinja2::ValuesMap extra_shape_value_map{{"indent", "    "}};
    std::string       extra_shape_func = TemplateLoadAndRender(extra_shape_template, extra_shape_value_map);

    jinja2::ValuesMap src_value_map{{"macro_decl", macro_decl},
                                    {"c_flag", "extern \"C\""},
                                    {"instances", instance_decl},
                                    {"function_name", func_name},
                                    {"shape_func", shape_eval_func},
                                    {"inverse_shape", inverse_shape},
                                    {"extra_shape", extra_shape_func},
                                    {"ndims", static_cast<int64_t>(ndims)},
                                    {"p_dims", p_dims},
                                    {"has_d0", has_d0_flag},
                                    {"has_d1", has_d1_flag},
                                    {"exec_paths", exec_paths},
                                    {"extra_code", extra_code},
                                    {"gemm_flag", gemm_flag},
                                    {"dtype_decl", dtype_decl},
                                    {"layout_decl", layout_decl},
                                    {"extra_header", extra_header},
                                    {"is_execute", true}};

    return TemplateLoadAndRender(g_src_source, src_value_map);
}

// void GemmCommonKernel::GemmCommonKernelLauncher(const std::string&        kernel_func_name,
//                                                 const GemmKernelArgs&     args,
//                                                 const GemmKernelCallType& kernel_call_type)
// {
//     VLOG(1) << args.GetDimInfo();

//     if (kernel_call_type == GemmKernelCallType::Gemm) {
//         decltype(&Gemm) kernel_func = nullptr;

//         LOAD_SYMBOL(kernel_func, kernel_func_name);

//         kernel_func(args.in_ptr_,
//                     args.weight_ptr_,
//                     args.out_ptr_,
//                     args.a_dim0_,
//                     args.a_dim1_,
//                     args.b_dim0_,
//                     args.b_dim1_,
//                     args.c_dim0_,
//                     args.c_dim1_,
//                     args.stream_);
//     }
//     else if (kernel_call_type == GemmKernelCallType::GemmBias) {

//         decltype(&GemmBias) kernel_func = nullptr;

//         LOAD_SYMBOL(kernel_func, kernel_func_name);

//         kernel_func(args.in_ptr_,
//                     args.weight_ptr_,
//                     args.out_ptr_,
//                     args.bias_ptr_,
//                     args.a_dim0_,
//                     args.a_dim1_,
//                     args.b_dim0_,
//                     args.b_dim1_,
//                     args.c_dim0_,
//                     args.c_dim1_,
//                     args.stream_);
//     }
//     else if (kernel_call_type == GemmKernelCallType::GemmBiasPermute) {
//         decltype(&GemmBiasPermute) kernel_func = nullptr;

//         LOAD_SYMBOL(kernel_func, kernel_func_name);

//         kernel_func(args.in_ptr_,
//                     args.weight_ptr_,
//                     args.out_ptr_,
//                     args.bias_ptr_,
//                     args.a_dim0_,
//                     args.a_dim1_,
//                     args.b_dim0_,
//                     args.b_dim1_,
//                     args.c_dim0_,
//                     args.c_dim1_,
//                     args.p_dim0_,
//                     args.p_dim1_,
//                     args.p_dim2_,
//                     args.stream_);
//     }
//     else if (kernel_call_type == GemmKernelCallType::GemmBiasElementwise) {
//         decltype(&GemmBiasElementwise) kernel_func = nullptr;

//         LOAD_SYMBOL(kernel_func, kernel_func_name);

//         kernel_func(args.in_ptr_,
//                     args.weight_ptr_,
//                     args.out_ptr_,
//                     args.bias_ptr_,
//                     args.d0_ptr_,
//                     args.a_dim0_,
//                     args.a_dim1_,
//                     args.b_dim0_,
//                     args.b_dim1_,
//                     args.c_dim0_,
//                     args.c_dim1_,
//                     args.stream_);
//     }
//     else {
//         LI_THROW(Unimplemented("unimplemented kernel_call_type: {}", static_cast<int>(kernel_call_type)));
//     }

//     // GpuDeviceSynchronize();

//     VLOG(1) << kernel_func_name << " kernel launch success";
// }
}  // namespace flashck