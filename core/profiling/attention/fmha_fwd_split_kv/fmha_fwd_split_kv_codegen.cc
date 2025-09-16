#include "core/profiling/attention/fmha_fwd_split_kv/fmha_fwd_split_kv_codegen.h"

namespace flashck {

std::string FmhaFwdSplitKVTileDesc::GetInstanceName() 
{
    return Sprintf(
        "{m0_block}x{n0_block}x{k0_block}_{n1_block}x{k1_block}x{k0_max_block}_r{m0_warp}x{n0_warp}x{k0_warp}_{m1_warp}x{n1_warp}x{k1_warp}_w{m0_warp}x{n0_warp_tile}x{k0_warp_tile}x{m1_warp_tile}x{n1_warp_tile}x{k1_warp_tile}",
        fmt::arg("m0_block", m0_block_),
        fmt::arg("n0_block", n0_block_),
        fmt::arg("k0_block", k0_block_),
        fmt::arg("n1_block", n1_block_),
        fmt::arg("k1_block", k1_block_),
        fmt::arg("k0_max_block", k0_max_block_),
        fmt::arg("m0_warp", m0_warp_),
        fmt::arg("n0_warp", n0_warp_),
        fmt::arg("k0_warp", k0_warp_),
        fmt::arg("m1_warp", m1_warp_),
        fmt::arg("n1_warp", n1_warp_),
        fmt::arg("k1_warp", k1_warp_),
        fmt::arg("m0_warp_tile", m0_warp_tile_),
        fmt::arg("n0_warp_tile", n0_warp_tile_),
        fmt::arg("k0_warp_tile", k0_warp_tile_),
        fmt::arg("m1_warp_tile", m1_warp_tile_),
        fmt::arg("n1_warp_tile", n1_warp_tile_),
        fmt::arg("k1_warp_tile", k1_warp_tile_));
}

std::string FmhaFwdSplitKVTileDesc::Emit() 
{
    std::string       tpl = R"(
    ck_tile::TileFmhaShape<ck_tile::sequence<{{m0_block}}, {{n0_block}}, {{k0_block}}, {{n1_block}}, {{k1_block}}, {{k0_max_block}}>,
                            ck_tile::sequence<{{m0_warp}}, {{n0_warp}}, {{k0_warp}}>,
                            ck_tile::sequence<{{m0_warp}}, {{n0_warp_tile}}, {{k0_warp_tile}}>,
                            ck_tile::sequence<{{m1_warp}}, {{n1_warp}}, {{k1_warp}}>,
                            ck_tile::sequence<{{m1_warp_tile}}, {{n1_warp_tile}}, {{k1_warp_tile}}>,
                            true /*kIsVLayoutRowMajor_*/ >
)";
    jinja2::ValuesMap value_map{{"m0_block", m0_block_},
                                {"n0_block", n0_block_},
                                {"k0_block", k0_block_},
                                {"n1_block", n1_block_},
                                {"k1_block", k1_block_},
                                {"k0_max_block", k0_max_block_},
                                {"m0_warp", m0_warp_},
                                {"n0_warp", n0_warp_},
                                {"k0_warp", k0_warp_},
                                {"m1_warp", m1_warp_},
                                {"n1_warp", n1_warp_},
                                {"k1_warp", k1_warp_},
                                {"m0_warp", m0_warp_},
                                {"n0_warp_tile", n0_warp_tile_},
                                {"k0_warp_tile", k0_warp_tile_},
                                {"m1_warp_tile", m1_warp_tile_},
                                {"n1_warp_tile", n1_warp_tile_},
                                {"k1_warp_tile", k1_warp_tile_}};
    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdTileDesc::Emit");
}


std::string FmhaFwdSplitKVCodeGen::GetPaddingConfigName() 
{
    return Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "kvs" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""));
}

std::string FmhaFwdSplitKVCodeGen::GetInstanceName() 
{
    auto trait = Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}_{has_uneven_splits}_{merge_groups_num_head_q_seq_len}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "sk" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""),
                   fmt::arg("has_uneven_splits", has_uneven_splits_ ? "u" : ""),
                   fmt::arg("merge_groups_num_head_q_seq_len", merge_groups_num_head_q_seq_len_ ? "m" : ""));

    auto strategy = Sprintf("{pipeline}_{num_splits}",
                   fmt::arg("pipeline", GetFwdSplitKVPipelineShortName(pipeline_)),
                   fmt::arg("num_splits", num_splits_));
    
    auto launch = Sprintf("{max_thread_per_block}_{min_block_per_cu}",
                   fmt::arg("max_thread_per_block", max_thread_per_block_),
                   fmt::arg("min_block_per_cu", min_block_per_cu_));

    return Sprintf("fmha_fwd_splitkv_{problem}_{tile_shape}_{trait}_{strategy}_{launch}",
                   fmt::arg("problem", problem_.GetName()),
                   fmt::arg("tile_shape", tile_desc_.GetInstanceName()),
                   fmt::arg("trait", trait),
                   fmt::arg("strategy", strategy),
                   fmt::arg("launch", launch));
}

std::string FmhaFwdSplitKVCodeGen::Emit() 
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
    ck_tile::ComposedAttention<{{has_logits_soft_cap}} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>,
    ck_tile::TileFmhaFwdSplitKVTraits<{{is_pad_q_seq_len}},
                                        {{is_pad_kv_seq_len}},
                                        {{is_pad_qk_head_dim}},
                                        {{is_pad_v_head_dim}},
                                        {{has_logits_soft_cap}},
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
        {"mode", false},
        {"has_logits_soft_cap", problem_.has_logits_soft_cap_},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
        {"is_pad_v_head_dim", is_pad_v_head_dim_},
        {"attention_bias", GetBiasClassTag(problem_.bias_enum_)},
        {"is_store_lse", false},
        {"is_static_quant", problem_.is_static_quant_},
        {"is_paged_kv", problem_.paged_block_size_ > 0},
        {"has_uneven_splits", has_uneven_splits_},
        {"merge_groups_num_head_q_seq_len", merge_groups_num_head_q_seq_len_},
        {"block_per_cu", min_block_per_cu_},
        {"pipeline", GetFwdSplitKVPipelineClassTag(pipeline_)}};

    return TEMPLATE_CHECK(source, value_map, "FmhaFwdSplitKVCodeGen::Emit");
}

}  // namespace flashck