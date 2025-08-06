#include "core/module/kernels/attention_kernels/fmha_fwd_split_kv/fmha_fwd_split_kv_kernel.h"

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
FmhaFwdSplitKVKernel::CodeGenForTuning(const std::string&    model_name,
                                        const instance_map_t& instance_map,
                                        const std::string&    folder_name)
{
    // Extract common configuration from first instance (assumes all instances share these properties)
    auto           instance           = std::get<fmha_fwd_splitkv_codegen_map_t>(instance_map).begin()->second;
    std::string           mode                = GetFmhaModeName(instance.problem_.mode_);
    int64_t num_splits = instance.problem_.num_splits_;
    int64_t        bias_rank_info      = instance.problem_.bias_rank_info_;
    std::string bias = GetBiasName(instance.problem_.bias_enum_);

    jinja2::ValuesMap func_signature_value_map{{"function_name", "FmhaFwdSplitKV"}};
    std::string       func_signature =
        TEMPLATE_CHECK(g_fmha_fwd_splitkv_func_signature_tpl, func_signature_value_map);

    jinja2::ValuesMap func_call_value_map{{"function_name", "FmhaFwdSplitKV"},
                                          {"bias", bias},
                                          {"mode", mode},
                                          {"paged_block_size", paged_block_size},
                                          {"use_cache_batch_idx", use_cache_batch_idx},
                                          {"mask", g_generic_attention_mask_names_map.at(mask_enum)}};
    std::string       func_call = TEMPLATE_CHECK(g_fmha_fwd_splitkv_func_call_tpl, func_call_value_map);

    jinja2::ValuesMap prepare_args_value_map{{"mode", mode},
                                             {"paged_block_size", paged_block_size},
                                             {"bias", g_bias_enum_names_map.at(bias_enum)},
                                             {"bias_rank_info", bias_rank_info}};
    std::string prepare_args = TEMPLATE_CHECK(g_fmha_fwd_splitkv_prepare_args_tpl, prepare_args_value_map);

    jinja2::ValuesMap make_args_value_map{{"mode", mode},
                                          {"kernel_name", "FmhaInstance"},
                                          {"paged_block_size", paged_block_size},
                                          {"bias", bias},
                                          {"bias_rank_info", bias_rank_info}};
    std::string       make_args = TEMPLATE_CHECK(g_fmha_fwd_splitkv_make_args_tpl, make_args_value_map);

    jinja2::ValuesMap tensor_decl_value_map{{"paged_block_size", paged_block_size},
                                            {"use_cache_batch_idx", use_cache_batch_idx},
                                            {"num_splits", num_splits},
                                            {"bias", bias},
                                            {"bias_rank_info", bias_rank_info}};
    std::string       tensor_decl = TEMPLATE_CHECK(g_fmha_fwd_splitkv_tensor_decl_tpl, tensor_decl_value_map);

    jinja2::ValuesMap tensor_generate_value_map{{"init_method", FLAGS_FC_TUNING_INIT_METHOD},
                                                {"mode", mode},
                                                {"bias", bias},
                                                {"bias_rank_info", bias_rank_info},
                                                {"num_splits", num_splits},
                                                {"paged_block_size", paged_block_size},
                                                {"use_cache_batch_idx", use_cache_batch_idx},
                                                {"seed", FLAGS_FC_TUNING_SEED}};
    std::string       tenosr_generate =
        TEMPLATE_CHECK(g_fmha_fwd_splitkv_tensor_generate_tpl, tensor_generate_value_map);

    return CommonCodeGenForTuning(model_name,
                                    FmhaTuningTpl{g_fmha_fwd_splitkv_create_args_tpl,
                                       g_fmha_fwd_splitkv_args_parser_tpl,
                                       g_fmha_fwd_splitkv_args_decl_tpl,
                                       func_signature,
                                       func_call,
                                       prepare_args,
                                       make_args,
                                       tensor_decl,
                                       tenosr_generate},
                                    instance_map);
}

std::string FmhaFwdSplitKVKernel::CodeGenForRunning(const std::string&                        func_name,
                                               const std::string&                        model_name,
                                               const std::map<std::string, RunningItem>& running_infos,
                                               const instance_map_t&                     instance_map,
                                               const std::string&                        folder_name)
{
    return CommonCodeGenForRunning(func_name, 
                                    model_name, 
                                    running_infos, 
                                    instance_map,
                                    FmhaRunningTpl{g_fmha_fwd_splitkv_args_decl_tpl,
                                    g_fmha_fwd_splitkv_func_signature_tpl,
                                    g_fmha_fwd_splitkv_prepare_args_tpl,
                                    g_fmha_fwd_splitkv_make_args_tpl},
                                    folder_name);
}

void FmhaFwdSplitKVKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    decltype(&FmhaFwdSplitKV) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    auto fmha_fwd_splitkv_kernel_args = std::get<FmhaFwdSplitKVKernelArgs>(args);

    kernel_func(fmha_fwd_splitkv_kernel_args.q_ptr_,
                fmha_fwd_splitkv_kernel_args.k_ptr_,
                fmha_fwd_splitkv_kernel_args.v_ptr_,
                fmha_fwd_splitkv_kernel_args.bias_ptr_,
                fmha_fwd_splitkv_kernel_args.lse_acc_ptr_,
                fmha_fwd_splitkv_kernel_args.out_acc_ptr_,
                fmha_fwd_splitkv_kernel_args.q_seq_start_ptr_,
                fmha_fwd_splitkv_kernel_args.kv_seq_start_ptr_,
                fmha_fwd_splitkv_kernel_args.k_seq_len_ptr_,
                fmha_fwd_splitkv_kernel_args.block_table_ptr_,
                fmha_fwd_splitkv_kernel_args.cache_batch_idx_ptr_,
                fmha_fwd_splitkv_kernel_args.batch_,
                fmha_fwd_splitkv_kernel_args.q_seq_len_,
                fmha_fwd_splitkv_kernel_args.seqlen_k_,
                fmha_fwd_splitkv_kernel_args.q_num_heads_,
                fmha_fwd_splitkv_kernel_args.kv_num_heads_,
                fmha_fwd_splitkv_kernel_args.qk_head_dim_,
                fmha_fwd_splitkv_kernel_args.v_head_dim_,
                fmha_fwd_splitkv_kernel_args.q_max_seq_len_,
                fmha_fwd_splitkv_kernel_args.max_num_page_blocks_,
                fmha_fwd_splitkv_kernel_args.paged_block_size_,
                fmha_fwd_splitkv_kernel_args.scale_,
                fmha_fwd_splitkv_kernel_args.window_size_,
                fmha_fwd_splitkv_kernel_args.mask_type_,
                fmha_fwd_splitkv_kernel_args.stream_);
}

}  // namespace flashck