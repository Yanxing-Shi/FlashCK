#include "core/profiling/tile/fmha/fmha_fwd_splitkv_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string FmhaFwdSplitKVCodeGen::GetPadName() const
{
    return Sprintf("p{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "kvs" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""));
}

std::string FmhaFwdSplitKVCodeGen::GetPipelineConfigName() const
{
    return Sprintf("{pad_name}_{bias_short_name}_{is_static_quant}{block_per_cu}",
                   fmt::arg("pad_name", GetPadName()),
                   fmt::arg("bias_short_name", GetBiasShortName(bias_enum_)),
                   fmt::arg("is_static_quant", is_static_quant_ ? "squant" : "nosquant"),
                   fmt::arg("block_per_cu", block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_)));
}

std::string FmhaFwdSplitKVCodeGen::GetInstanceName() const
{
    return Sprintf("fmha_fwd_splitkv_{dtype}_{mode}_{tile_desc}_{pipeline}_{mask}",
                   fmt::arg("dtype", DataTypeToString(dtype_)),
                   fmt::arg("mode", GetFmhaModeName(mode_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline", GetPipelineConfigName()),
                   fmt::arg("mask", GetAttentionMaskShortName(mask_type_)));
}

std::string FmhaFwdSplitKVCodeGen::Emit() const
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
        {"name", GetInstanceName()},
        {"idx", idx++},
        {"shape", tile_desc_.Emit()},
        {"mode", mode_ == FmhaMode::Batch ? "false" : "true"},
        {"mask",
         mask_type_ == GenericAttentionMaskEnum::NO_MASK || (window_size_[0] == -1 && window_size_[1] == -1) ? "false" :
                                                                                                               "true"},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
        {"is_pad_v_head_dim", is_pad_v_head_dim_},
        {"attention_bias", GetBiasClassTag(bias_enum_)},
        {"is_store_lse", is_store_lse_},
        {"is_static_quant", is_static_quant_},
        {"is_paged_kv", is_paged_kv_},
        {"has_uneven_splits", has_uneven_splits_},
        {"is_merge_num_head_groups_seq_len_q", is_merge_num_head_groups_seq_len_q_},
        {"block_per_cu", std::to_string(block_per_cu_)},
        {"pipeline", GetFwdSplitKVPipelineClassTag(pipeline_)}};

    return TEMPLATE_CHECK(source, value_map, "FmhaFwdSplitKVCodeGen::Emit");
}

}  // namespace flashck