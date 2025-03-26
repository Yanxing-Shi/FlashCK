#include "lightinfer/core/module/kernels/fmha_kernels/fmha_fwd_splitkv_kernel.h"

#include "lightinfer/core/utils/jinjia2_utils.h"

namespace lightinfer {

std::map<std::string, std::shared_ptr<void>> FmhaFwdSplitKVKernel::Init(const OperationKind&   op_kind,
                                                                        const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<FmhaOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
FmhaFwdSplitKVKernel::GenKernelProfiler(const std::string&                               model_name,
                                        const std::unordered_map<std::string, std::any>& kernel_func_map,
                                        const std::string&                               folder_name)
{
    auto op_mode             = std::any_cast<FmhaOperationMode>(kernel_func_map.at("op_mode"));
    auto num_splits          = std::any_cast<int64_t>(kernel_func_map.at("num_splits"));
    auto use_cache_batch_idx = std::any_cast<bool>(kernel_func_map.at("use_cache_batch_idx"));
    auto paged_block_size    = std::any_cast<int64_t>(kernel_func_map.at("paged_block_size"));
    auto bias_enum           = std::any_cast<BiasEnum>(kernel_func_map.at("bias_enum"));
    auto bias_rank_info      = std::any_cast<int64_t>(kernel_func_map.at("bias_rank_info"));
    auto mask_enum           = std::any_cast<GenericAttentionMaskEnum>(kernel_func_map.at("mask_enum"));

    jinja2::ValuesMap func_signature_value_map{{"function_name", "FmhaFwdSplitKV"}, {"is_execute", false}};
    std::string       func_signature =
        TemplateLoadAndRender(g_fmha_fwd_splitkv_func_signature_source, func_signature_value_map);

    jinja2::ValuesMap func_call_value_map{{"function_name", "FmhaFwdSplitKV"},
                                          {"bias_str", g_bias_enum_names_map.at(bias_enum)},
                                          {"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                          {"paged_block_size", paged_block_size},
                                          {"use_cache_batch_idx", use_cache_batch_idx},
                                          {"mask_str", g_generic_attention_mask_names_map.at(mask_enum)},
                                          {"num_splits", num_splits}};
    std::string       func_call = TemplateLoadAndRender(g_fmha_fwd_splitkv_func_call_source, func_call_value_map);

    jinja2::ValuesMap prepare_args_value_map{{"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                             {"paged_block_size", paged_block_size},
                                             {"bias_str", g_bias_enum_names_map.at(bias_enum)},
                                             {"bias_rank_info", bias_rank_info}};
    std::string prepare_args = TemplateLoadAndRender(g_fmha_fwd_splitkv_prepare_args_source, prepare_args_value_map);

    jinja2::ValuesMap make_args_value_map{{"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                          {"kernel_name", "FmhaInstance"},
                                          {"is_execute", false},
                                          {"paged_block_size", paged_block_size},
                                          {"bias_str", g_bias_enum_names_map.at(bias_enum)},
                                          {"bias_rank_info", bias_rank_info}};
    std::string       make_args = TemplateLoadAndRender(g_fmha_fwd_splitkv_make_args_source, make_args_value_map);

    jinja2::ValuesMap tensor_decl_value_map{{"paged_block_size", paged_block_size},
                                            {"use_cache_batch_idx", use_cache_batch_idx},
                                            {"num_splits", num_splits},
                                            {"bias_str", g_bias_enum_names_map.at(bias_enum)},
                                            {"bias_rank_info", bias_rank_info}};
    std::string       tensor_decl = TemplateLoadAndRender(g_fmha_fwd_splitkv_tensor_decl_source, tensor_decl_value_map);

    jinja2::ValuesMap tensor_generate_value_map{{"init_method_str", "uf"},
                                                {"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                                {"bias_str", g_bias_enum_names_map.at(bias_enum)},
                                                {"bias_rank_info", bias_rank_info},
                                                {"num_splits", num_splits},
                                                {"paged_block_size", paged_block_size},
                                                {"use_cache_batch_idx", use_cache_batch_idx},
                                                {"seed", 12456}};
    std::string       tenosr_generate =
        TemplateLoadAndRender(g_fmha_fwd_splitkv_tensor_generate_source, tensor_generate_value_map);

    return GenFmhaCommonKernelProfiler(model_name,
                                       kernel_func_map,
                                       g_fmha_fwd_splitkv_create_args_source,
                                       g_fmha_fwd_splitkv_args_parser_source,
                                       g_fmha_fwd_splitkv_args_decl_source,
                                       func_signature,
                                       tensor_decl,
                                       tenosr_generate,
                                       prepare_args,
                                       func_call,
                                       make_args,
                                       "fwd_splitkv");
}

std::string FmhaFwdSplitKVKernel::GenKernelFunction(const std::string&                               func_name,
                                                    const std::string&                               model_name,
                                                    const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenFmhaCommonKernelFunction(func_name,
                                       model_name,
                                       kernel_func_map,
                                       g_fmha_fwd_splitkv_args_decl_source,
                                       g_fmha_fwd_splitkv_func_signature_source,
                                       g_fmha_fwd_splitkv_prepare_args_source,
                                       g_fmha_fwd_splitkv_make_args_source,
                                       "fwd_splitkv");
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
                fmha_fwd_splitkv_kernel_args.seqstart_q_ptr_,
                fmha_fwd_splitkv_kernel_args.seqstart_k_ptr_,
                fmha_fwd_splitkv_kernel_args.seqlen_k_ptr_,
                fmha_fwd_splitkv_kernel_args.block_table_ptr_,
                fmha_fwd_splitkv_kernel_args.cache_batch_idx_ptr_,
                fmha_fwd_splitkv_kernel_args.batch_,
                fmha_fwd_splitkv_kernel_args.seqlen_q_,
                fmha_fwd_splitkv_kernel_args.seqlen_k_,
                fmha_fwd_splitkv_kernel_args.nhead_q_,
                fmha_fwd_splitkv_kernel_args.nhead_k_,
                fmha_fwd_splitkv_kernel_args.hdim_q_,
                fmha_fwd_splitkv_kernel_args.hdim_v_,
                fmha_fwd_splitkv_kernel_args.max_seqlen_q_,
                fmha_fwd_splitkv_kernel_args.max_num_page_blocks_,
                fmha_fwd_splitkv_kernel_args.paged_block_size_,
                fmha_fwd_splitkv_kernel_args.scale_,
                fmha_fwd_splitkv_kernel_args.window_size_,
                fmha_fwd_splitkv_kernel_args.mask_type_,
                fmha_fwd_splitkv_kernel_args.stream_);
}

}  // namespace lightinfer