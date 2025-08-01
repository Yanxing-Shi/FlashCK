#include "core/module/kernels/gemm_kernels/gemm_multi_d/gemm_multi_d_kernel.h"


namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmMultiD::CodeGenForTuning(const std::string&    model_name,
                                const instance_map_t& instance_map,
                                const std::string&    folder_name)
{
    auto           instance           = std::get<gemm_codegen_map_t>(instance_map).begin()->second;
    int64_t        tensor_num_d      = instance.problem_.ds_layout_.size();

    std::string header = TEMPLATE_CHECK(g_flat_gemm_header_tpl, {{}});

    std::string create_args = TEMPLATE_CHECK(g_flat_gemm_create_args_tpl, {{}});

    std::string arg_parser = TEMPLATE_CHECK(g_flat_gemm_args_parser_tpl, {{}});

    jinja2::ValuesMap make_args_value_map{{"tensor_num_d", tensor_num_d}};
    std::string       make_args = TEMPLATE_CHECK(g_flat_gemm_make_args_tpl, make_args_value_map);

    jinja2::ValuesMap func_signature_value_map{{"function_name", "GemmMultiD"},{"tensor_num_d", tensor_num_d}};
    std::string func_signature = TEMPLATE_CHECK(g_flat_gemm_func_signature_tpl, func_signature_value_map);

    jinja2::ValuesMap func_call_value_map{{"function_name", "GemmMultiD"}};
    std::string       func_call = TEMPLATE_CHECK(g_flat_gemm_func_call_tpl, func_call_value_map);

    jinja2::ValuesMap tensor_decl_value_map{{"tensor_num_d", tensor_num_d}};
    std::string       tensor_decl = TEMPLATE_CHECK(g_flat_gemm_tensor_decl_tpl, tensor_decl_value_map);

    return CommonCodeGenForTuning(model_name,
                                GemmTuningTpl{
                                header,
                                create_args,
                                arg_parser,
                                make_args,
                                func_signature,
                                func_call,
                                tensor_decl,
                                },
                                instance_map);
}

std::string GemmMultiD::CodeGenForRunning(const std::string&                        func_name,
                                               const std::string&                        model_name,
                                               const std::map<std::string, RunningItem>& running_infos,
                                               const instance_map_t&                     instance_map,
                                               const std::string&                        folder_name)
{
    auto           instance           = std::get<gemm_codegen_map_t>(instance_map).begin()->second;
    int64_t        tensor_num_d      = instance.problem_.ds_layout_.size();

    std::string header = TEMPLATE_CHECK(g_flat_gemm_header_tpl, {{}});

    jinja2::ValuesMap make_args_value_map{{"tensor_num_d", tensor_num_d}};
    std::string       make_args = TEMPLATE_CHECK(g_flat_gemm_make_args_tpl, make_args_value_map);
    
    jinja2::ValuesMap func_signature_value_map{{"function_name", func_name},{"tensor_num_d", tensor_num_d}};
    std::string func_signature = TEMPLATE_CHECK(g_flat_gemm_func_signature_tpl, func_signature_value_map);
    
    return  CommonCodeGenForRunning(func_name, 
                                    model_name, 
                                    running_infos, 
                                    instance_map, 
                                    GemmRunningTpl{
                                    header,
                                    make_args,
                                    func_signature}, 
                                    folder_name);
}

void GemmMultiD::KernelLauncher(const std::string& kernel_func_name, const KernelArgs_t& args)
{
    decltype(&FmhaFwd) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    auto gemm_multi_d_kernel_args = std::get<GemmMultiDKernelArgs<1>>(args);

    kernel_func(gemm_multi_d_kernel_args.a_ptr_,
                gemm_multi_d_kernel_args.b_ptr_,
                gemm_multi_d_kernel_args.ds_ptr_,
                gemm_multi_d_kernel_args.c_ptr_,
                gemm_multi_d_kernel_args.split_k_,
                gemm_multi_d_kernel_args.m_,
                gemm_multi_d_kernel_args.n_,
                gemm_multi_d_kernel_args.k_,
                gemm_multi_d_kernel_args.a_stride_,
                gemm_multi_d_kernel_args.b_stride_,
                gemm_multi_d_kernel_args.ds_stride_,
                gemm_multi_d_kernel_args.c_stride_,
                gemm_multi_d_kernel_args.stream_);
}

}  // namespace flashck