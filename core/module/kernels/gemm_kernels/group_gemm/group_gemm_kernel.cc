#include "core/module/kernels/gemm_kernels/gemm/batch_gemm_kernel.h"


namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GroupGemmKernel::CodeGenForTuning(const std::string&    model_name,
                             const instance_map_t& instance_map,
                             const std::string&    folder_name)
{
    std::string header = TEMPLATE_CHECK(g_group_gemm_header_tpl, {{}});

    std::string create_args = TEMPLATE_CHECK(g_group_gemm_create_args_tpl, {{}});

    std::string arg_parser = TEMPLATE_CHECK(g_group_gemm_args_parser_tpl, {{}});

    std::string make_args = TEMPLATE_CHECK(g_group_gemm_make_args_tpl, {{}});

    jinja2::ValuesMap func_signature_value_map{{"function_name", "Gemm"}};
    std::string func_signature = TEMPLATE_CHECK(g_group_gemm_func_signature_tpl, func_signature_value_map);

    jinja2::ValuesMap func_call_value_map{{"function_name", "Gemm"}};
    std::string       func_call = TEMPLATE_CHECK(g_group_gemm_func_call_tpl, func_call_value_map);

    std::string       tensor_decl = TEMPLATE_CHECK(g_group_gemm_tensor_decl_tpl, {{}});

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

std::string GroupGemmKernel::CodeGenForRunning(const std::string&                        func_name,
                                               const std::string&                        model_name,
                                               const std::map<std::string, RunningItem>& running_infos,
                                               const instance_map_t&                     instance_map,
                                               const std::string&                        folder_name)
{
    std::string header = TEMPLATE_CHECK(g_group_gemm_header_tpl, {{}});

    std::string       make_args = TEMPLATE_CHECK(g_group_gemm_make_args_tpl, {{}});

    jinja2::ValuesMap func_signature_value_map{{"function_name", func_name}};
    std::string func_signature = TEMPLATE_CHECK(g_group_gemm_func_signature_tpl, func_signature_value_map);
    
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

void GroupGemmKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs_t& args)
{
    decltype(&Gemm) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    auto group_gemm_kernel_args = std::get<GroupGroupGemmKernelArgs>(args);

    kernel_func(group_gemm_kernel_args.a_ptr_,
                group_gemm_kernel_args.b_ptr_,
                group_gemm_kernel_args.c_ptr_,
                group_gemm_kernel_args.split_k_,
                group_gemm_kernel_args.m_,
                group_gemm_kernel_args.n_,
                group_gemm_kernel_args.k_,
                group_gemm_kernel_args.a_stride_,
                group_gemm_kernel_args.b_stride_,
                group_gemm_kernel_args.c_stride_,
                group_gemm_kernel_args.stream_);
}

}  // namespace flashck