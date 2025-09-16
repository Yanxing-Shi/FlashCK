#include "core/module/kernels/attention_kernels/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_kernel.h"

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
FmhaFwdSplitKVCombineKernel::GenKernelProfiler(const std::string&                               model_name,
                                               const std::unordered_map<std::string, std::any>& kernel_func_map,
                                               const std::string&                               folder_name)
{
    // Extract common configuration from first instance (assumes all instances share these properties)
    auto           instance           = std::get<fmha_fwd_codegen_map_t>(instance_map).begin()->second;
    std::string           mode                = GetFmhaModeName(instance.problem_.mode_);
    int64_t        bias_rank_info      = instance.problem_.bias_rank_info_;
    std::string bias = GetBiasName(instance.problem_.bias_enum_);

    jinja2::ValuesMap func_signature_value_map{{"function_name", "FmhaSplitKVCombine"}};
    std::string       func_signature =
        TEMPLATE_CHECK(g_fmha_fwd_splitkv_combine_func_signature_tpl, func_signature_value_map);

    jinja2::ValuesMap func_call_value_map{{"function_name", "FmhaSplitKVCombine"},
                                          {"mode", g_fmha_operation_mode_name_map.at(op_mode)}};
    std::string func_call = TEMPLATE_CHECK(g_fmha_fwd_splitkv_combine_func_call_tpl, func_call_value_map);

    jinja2::ValuesMap prepare_args_value_map{{"mode", g_fmha_operation_mode_name_map.at(op_mode)}};
    std::string       prepare_args =
        TEMPLATE_CHECK(g_fmha_fwd_splitkv_combine_prepare_args_tpl, prepare_args_value_map);

    jinja2::ValuesMap make_args_value_map{{"mode", g_fmha_operation_mode_name_map.at(op_mode)},
                                          {"kernel_name", "FmhaInstance"}};
    std::string make_args = TEMPLATE_CHECK(g_fmha_fwd_splitkv_combine_make_args_tpl, make_args_value_map);

    jinja2::ValuesMap tensor_decl_value_map{{"num_splits", num_splits}};
    std::string       tensor_decl =
        TEMPLATE_CHECK(g_fmha_fwd_splitkv_combine_tensor_decl_tpl, tensor_decl_value_map);

    jinja2::ValuesMap tensor_generate_value_map{{"num_splits", num_splits}, {"seed", 12456}};
    std::string       tensor_generate =
        TEMPLATE_CHECK(g_fmha_fwd_splitkv_combine_tensor_generate_tpl, tensor_generate_value_map);

    return CommonCodeGenForTuning(model_name,
                                       FmhaTuningTpl{g_fmha_fwd_splitkv_combine_create_args_tpl,
                                       g_fmha_fwd_splitkv_combine_args_parser_tpl,
                                       g_fmha_fwd_splitkv_combine_args_decl_tpl,
                                       func_signature,
                                       func_call,
                                       prepare_args,
                                       make_args,
                                       tensor_decl,
                                       tenosr_generate});
}

std::string
FmhaFwdSplitKVCombineKernel::CodeGenForRunning(const std::string&                        func_name,
                                               const std::string&                        model_name,
                                               const std::map<std::string, RunningItem>& running_infos,
                                               const instance_map_t&                     instance_map,
                                               const std::string&                        folder_name)
{
    return CommonCodeGenForRunning(func_name,
                                       model_name,
                                       kernel_func_map,
                                       FmhaRunningTpl{g_fmha_fwd_splitkv_combine_args_decl_tpl,
                                       g_fmha_fwd_splitkv_combine_func_signature_tpl,
                                       g_fmha_fwd_splitkv_combine_prepare_args_tpl,
                                       g_fmha_fwd_splitkv_combine_make_args_tpl},
                                       folder_name);
}

void FmhaFwdSplitKVCombineKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    decltype(&FmhaFwdSplitKVCombine) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    auto fmha_fwd_splitkv_combine_kernel_args = std::get<FmhaFwdSplitKVCombineKernelArgs>(args);

    kernel_func(fmha_fwd_splitkv_combine_kernel_args.lse_acc_ptr_,
                fmha_fwd_splitkv_combine_kernel_args.out_acc_ptr_,
                fmha_fwd_splitkv_combine_kernel_args.out_ptr_,
                fmha_fwd_splitkv_combine_kernel_args.q_seq_start_ptr_,
                fmha_fwd_splitkv_combine_kernel_args.batch_,
                fmha_fwd_splitkv_combine_kernel_args.q_seq_len_,
                fmha_fwd_splitkv_combine_kernel_args.q_num_heads_,
                fmha_fwd_splitkv_combine_kernel_args.v_head_dim_,
                fmha_fwd_splitkv_combine_kernel_args.q_max_seq_len_,
                fmha_fwd_splitkv_combine_kernel_args.num_splits_,
                fmha_fwd_splitkv_combine_kernel_args.stream_);
}

}  // namespace flashck