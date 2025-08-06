#include "core/module/kernels/attention_kernels/fmha_fwd_append_kv/fmha_fwd_append_kv_kernel.h"

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
FmhaFwdAppendKVKernel::CodeGenForTuning(const std::string&                               model_name,
                                         const std::unordered_map<std::string, std::any>& kernel_func_map,
                                         const std::string&                               folder_name)
{

    jinja2::ValuesMap func_signature_value_map{{"function_name", "FmhaFwdAppendKV"}, {"is_execute", false}};
    std::string       func_signature =
        TemplateLoadAndRender(g_fmha_fwd_appendkv_func_signature_tpl, func_signature_value_map);

    jinja2::ValuesMap func_call_value_map{{"function_name", "FmhaFwdAppendKV"},
                                          {"paged_block_size", paged_block_size},
                                          {"use_cache_batch_idx", use_cache_batch_idx},
                                          {"mask", g_generic_attention_mask_names_map.at(mask_enum)}};
    std::string       func_call = TemplateLoadAndRender(g_fmha_fwd_appendkv_func_call_tpl, func_call_value_map);

    jinja2::ValuesMap prepare_args_value_map{{"mode", g_fmha_operation_mode_name_map.at(op_mode)}};
    std::string prepare_args = TemplateLoadAndRender(g_fmha_fwd_appendkv_prepare_args_tpl, prepare_args_value_map);

    jinja2::ValuesMap make_args_value_map{{"kernel_name", "FmhaInstance"}};
    std::string       make_args = TemplateLoadAndRender(g_fmha_fwd_appendkv_make_args_tpl, make_args_value_map);

    jinja2::ValuesMap tensor_decl_value_map{{"paged_block_size", paged_block_size},
                                            {"use_cache_batch_idx", use_cache_batch_idx}};
    std::string tensor_decl = TemplateLoadAndRender(g_fmha_fwd_appendkv_tensor_decl_tpl, tensor_decl_value_map);

    jinja2::ValuesMap tensor_generate_value_map{{"init_method", "uf"},
                                                {"mode", g_fmha_operation_mode_name_map.at(op_mode)},
                                                {"rotary_dim", rotary_dim},
                                                {"paged_block_size", paged_block_size},
                                                {"use_cache_batch_idx", use_cache_batch_idx},
                                                {"seed", 12456}};
    std::string       tenosr_generate =
        TemplateLoadAndRender(g_fmha_fwd_appendkv_tensor_generate_tpl, tensor_generate_value_map);

    return GenFmhaCommonKernelProfiler(model_name,
                                       FmhaTuningTpl{g_fmha_fwd_appendkv_create_args_tpl,
                                       g_fmha_fwd_appendkv_args_parser_tpl,
                                       g_fmha_fwd_appendkv_args_decl_tpl,
                                       func_signature,
                                       func_call,
                                       prepare_args,
                                       make_args,
                                       tensor_decl,
                                       tensor_generate},
                                       folder_name);
}

std::string FmhaFwdAppendKVKernel::CodeGenForRunning(const std::string&                        func_name,
                                               const std::string&                        model_name,
                                               const std::map<std::string, RunningItem>& running_infos,
                                               const instance_map_t&                     instance_map,
                                               const std::string&                        folder_name)
{
    return GenFmhaCommonKernelFunction(func_name,
                                       model_name,
                                       running_infos, 
                                       instance_map, 
                                       FmhaRunningTpl{g_fmha_fwd_appendkv_args_decl_tpl,
                                       g_fmha_fwd_appendkv_func_signature_tpl,
                                       g_fmha_fwd_appendkv_prepare_args_tpl,
                                       g_fmha_fwd_appendkv_make_args_tpl},
                                       folder_name);
}

void FmhaFwdAppendKVKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    decltype(&FmhaFwdAppendKV) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    auto fmha_fwd_appendkv_kernel_args = std::get<FmhaFwdAppendKVKernelArgs>(args);

    kernel_func(fmha_fwd_appendkv_kernel_args.q_ptr_,
                fmha_fwd_appendkv_kernel_args.k_cache_ptr_,
                fmha_fwd_appendkv_kernel_args.v_cache_ptr_,
                fmha_fwd_appendkv_kernel_args.k_ptr_,
                fmha_fwd_appendkv_kernel_args.v_ptr_,
                fmha_fwd_appendkv_kernel_args.k_cache_seq_len_ptr_,
                fmha_fwd_appendkv_kernel_args.rotary_cos_ptr_,
                fmha_fwd_appendkv_kernel_args.rotary_sin_ptr_,
                fmha_fwd_appendkv_kernel_args.block_table_ptr_,
                fmha_fwd_appendkv_kernel_args.cache_batch_idx_ptr_,
                fmha_fwd_appendkv_kernel_args.batch_,
                fmha_fwd_appendkv_kernel_args.q_seq_len_,
                fmha_fwd_appendkv_kernel_args.kv_seq_len_,
                fmha_fwd_appendkv_kernel_args.q_num_heads_,
                fmha_fwd_appendkv_kernel_args.kv_num_heads_,
                fmha_fwd_appendkv_kernel_args.qk_head_dim_,
                fmha_fwd_appendkv_kernel_args.v_head_dim_,
                fmha_fwd_appendkv_kernel_args.new_k_seq_len,
                fmha_fwd_appendkv_kernel_args.max_num_page_blocks_,
                fmha_fwd_appendkv_kernel_args.paged_block_size_,
                fmha_fwd_appendkv_kernel_args.rotary_dim_,
                fmha_fwd_appendkv_kernel_args.has_mask_,
                fmha_fwd_appendkv_kernel_args.stream_);
}

}  // namespace flashck