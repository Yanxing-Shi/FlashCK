#include "flashck/core/profiling/fmha_fwd_splitkv_operation.h"

#include "flashck/core/utils/jinjia2_utils.h"

namespace flashck {

std::string FmhaFwdSplitKVOperation::GetPadName()
{
    std::ostringstream oss;
    oss << "p" << (is_pad_q_seq_len_ ? "s" : "") << (is_pad_kv_seq_len_ ? "kvs" : "")
        << (is_pad_qk_head_dim_ ? "d" : "") << (is_pad_v_head_dim_ ? "dv" : "");
    return oss.str();
}

std::string FmhaFwdSplitKVOperation::GetPipelineConfigName()
{
    std::ostringstream oss;
    oss << GetPadName() << "_" << g_bias_enum_short_names_map.at(bias_enum_) << "_"
        << (is_static_quant_ ? "squant" : "nosquant")
        << (block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_));
    return oss.str();
}

std::string FmhaFwdSplitKVOperation::GetConfigName()
{
    std::ostringstream oss;
    oss << "fmha_fwd_splitkv" << "_" << DataTypeToShortString(dtype_) << "_"
        << g_fmha_operation_mode_name_map.at(operation_mode_) << "_" << tile_desc_.GetConfigName() << "_"
        << GetPipelineConfigName() << "_" << g_generic_attention_mask_short_names_map.at(mask_type_);
    return oss.str();
}

std::string FmhaFwdSplitKVOperation::Emit()
{
    std::string source = R"(
using fmha_fwd_splitkv_pipeline_problem_{{idx}} = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    QDataType,
    KDataType,
    VDataType,
    SaccDataType,
    SMPLComputeDataType,
    BiasDataType,
    LSEDataType,
    PDataType,
    OaccDataType,
    ODataType,
    {{shape}},
    {{mode}},
    ck_tile::SimplifiedGenericAttentionMask<{{mask}}>,
    ck_tile::TileFmhaFwdSplitKVTraits<{{is_pad_q_seq_len}},
                                        {{is_pad_kv_seq_len}},
                                        {{is_pad_qk_head_dim}},
                                        {{is_pad_v_head_dim}},
                                        {{attention_bias}},
                                        false, // kHasBiasGrad
                                        {{is_store_lse}},
                                        {{is_static_quant}},
                                        {{is_paged_kv}},
                                        {{has_uneven_splits}},
                                        {{is_merge_num_head_groups_seq_len_q}},
                                        {{block_per_cu}}>>;

using {{name}} = 
    ck_tile::FmhaFwdSplitKVKernel<{{pipeline}}<fmha_fwd_splitkv_pipeline_problem_{{idx}}>,
                                    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<OaccDataType, ODataType,
                                                    {{is_pad_q_seq_len}}, {{is_pad_v_head_dim}}>>>;

)";
    static int  idx    = 0;

    jinja2::ValuesMap value_map = {
        {"name", GetConfigName()},
        {"idx", idx++},
        {"shape", tile_desc_.Emit()},
        {"mode", operation_mode_ == FmhaOperationMode::Batch ? "false" : "true"},
        {"mask",
         mask_type_ == GenericAttentionMaskEnum::NO_MASK || (window_size_[0] == -1 && window_size_[1] == -1) ? "false" :
                                                                                                               "true"},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qkv_head_dim_},
        {"is_pad_v_head_dim", is_pad_qkv_head_dim_},
        {"attention_bias", g_bias_enum_tag.at(bias_enum_)},
        {"is_store_lse", is_store_lse_},
        {"is_static_quant", is_static_quant_},
        {"is_paged_kv", is_paged_kv_},
        {"has_uneven_splits", has_uneven_splits_},
        {"is_merge_num_head_groups_seq_len_q", is_merge_num_head_groups_seq_len_q_},
        {"block_per_cu", block_per_cu_},
        {"pipeline", g_block_fmha_fwd_splitkv_pipeline_map.at(pipeline_)}};

    return TemplateLoadAndRender(source, value_map);
}

}  // namespace flashck