#include "flashck/core/module/kernels/fmha_kernels/fmha_fwd_appendkv_kernel.h"

#include "flashck/core/utils/jinjia2_utils.h"

namespace flashck {

std::map<std::string, std::shared_ptr<void>> FmhaFwdAppendKVKernel::Init(const OperationKind&   op_kind,
                                                                         const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<FmhaOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
FmhaFwdAppendKVKernel::GenKernelProfiler(const std::string&                               model_name,
                                         const std::unordered_map<std::string, std::any>& kernel_func_map,
                                         const std::string&                               folder_name)
{
    auto op_mode             = std::any_cast<FmhaOperationMode>(kernel_func_map.at("op_mode"));
    auto paged_block_size    = std::any_cast<int64_t>(kernel_func_map.at("paged_block_size"));
    auto use_cache_batch_idx = std::any_cast<bool>(kernel_func_map.at("use_cache_batch_idx"));
    auto rotary_dim          = std::any_cast<int64_t>(kernel_func_map.at("rotary_dim"));
    auto mask_enum           = std::any_cast<GenericAttentionMaskEnum>(kernel_func_map.at("mask_enum"));

    jinja2::ValuesMap func_signature_value_map{{"function_name", "FmhaFwdAppendKV"}, {"is_execute", false}};
    std::string       func_signature =
        TemplateLoadAndRender(g_fmha_fwd_appendkv_func_signature_source, func_signature_value_map);

    jinja2::ValuesMap func_call_value_map{{"function_name", "FmhaFwdAppendKV"},
                                          {"paged_block_size", paged_block_size},
                                          {"use_cache_batch_idx", use_cache_batch_idx},
                                          {"mask_str", g_generic_attention_mask_names_map.at(mask_enum)}};
    std::string       func_call = TemplateLoadAndRender(g_fmha_fwd_appendkv_func_call_source, func_call_value_map);

    jinja2::ValuesMap prepare_args_value_map{{"mode_str", g_fmha_operation_mode_name_map.at(op_mode)}};
    std::string prepare_args = TemplateLoadAndRender(g_fmha_fwd_appendkv_prepare_args_source, prepare_args_value_map);

    jinja2::ValuesMap make_args_value_map{{"kernel_name", "FmhaInstance"}};
    std::string       make_args = TemplateLoadAndRender(g_fmha_fwd_appendkv_make_args_source, make_args_value_map);

    jinja2::ValuesMap tensor_decl_value_map{{"paged_block_size", paged_block_size},
                                            {"use_cache_batch_idx", use_cache_batch_idx}};
    std::string tensor_decl = TemplateLoadAndRender(g_fmha_fwd_appendkv_tensor_decl_source, tensor_decl_value_map);

    jinja2::ValuesMap tensor_generate_value_map{{"init_method_str", "uf"},
                                                {"mode_str", g_fmha_operation_mode_name_map.at(op_mode)},
                                                {"rotary_dim", rotary_dim},
                                                {"paged_block_size", paged_block_size},
                                                {"use_cache_batch_idx", use_cache_batch_idx},
                                                {"seed", 12456}};
    std::string       tenosr_generate =
        TemplateLoadAndRender(g_fmha_fwd_appendkv_tensor_generate_source, tensor_generate_value_map);

    return GenFmhaCommonKernelProfiler(model_name,
                                       kernel_func_map,
                                       g_fmha_fwd_appendkv_create_args_source,
                                       g_fmha_fwd_appendkv_args_parser_source,
                                       g_fmha_fwd_appendkv_args_decl_source,
                                       func_signature,
                                       tensor_decl,
                                       tenosr_generate,
                                       prepare_args,
                                       func_call,
                                       make_args,
                                       "fwd_appendkv");
}

std::string FmhaFwdAppendKVKernel::GenKernelFunction(const std::string&                               func_name,
                                                     const std::string&                               model_name,
                                                     const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenFmhaCommonKernelFunction(func_name,
                                       model_name,
                                       kernel_func_map,
                                       g_fmha_fwd_appendkv_args_decl_source,
                                       g_fmha_fwd_appendkv_func_signature_source,
                                       g_fmha_fwd_appendkv_prepare_args_source,
                                       g_fmha_fwd_appendkv_make_args_source,
                                       "fwd_appendkv");
}

void FmhaFwdAppendKVKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    decltype(&FmhaFwdAppendKV) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    auto fmha_fwd_appendkv_kernel_args = std::get<FmhaFwdAppendKVKernelArgs>(args);

    kernel_func(fmha_fwd_appendkv_kernel_args.q_ptr_,
                fmha_fwd_appendkv_kernel_args.cache_k_ptr_,
                fmha_fwd_appendkv_kernel_args.cache_v_ptr_,
                fmha_fwd_appendkv_kernel_args.k_ptr_,
                fmha_fwd_appendkv_kernel_args.v_ptr_,
                fmha_fwd_appendkv_kernel_args.cache_seqlen_k_ptr_,
                fmha_fwd_appendkv_kernel_args.rotary_cos_ptr_,
                fmha_fwd_appendkv_kernel_args.rotary_sin_ptr_,
                fmha_fwd_appendkv_kernel_args.block_table_ptr_,
                fmha_fwd_appendkv_kernel_args.cache_batch_idx_ptr_,
                fmha_fwd_appendkv_kernel_args.batch_,
                fmha_fwd_appendkv_kernel_args.seqlen_q_,
                fmha_fwd_appendkv_kernel_args.seqlen_k_,
                fmha_fwd_appendkv_kernel_args.nhead_q_,
                fmha_fwd_appendkv_kernel_args.nhead_k_,
                fmha_fwd_appendkv_kernel_args.hdim_q_,
                fmha_fwd_appendkv_kernel_args.hdim_v_,
                fmha_fwd_appendkv_kernel_args.seqlen_knew_,
                fmha_fwd_appendkv_kernel_args.max_num_page_blocks_,
                fmha_fwd_appendkv_kernel_args.paged_block_size_,
                fmha_fwd_appendkv_kernel_args.rotary_dim_,
                fmha_fwd_appendkv_kernel_args.has_mask_,
                fmha_fwd_appendkv_kernel_args.stream_);
}

}  // namespace flashck