#include "core/profiling/tile/fmha/fmha_fwd_split_kv_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string FmhaFwdSplitKVTileDesc::GetInstanceName() const
{
    return Sprintf(
        "{m0_block}x{n0_block}x{k0_block}_{n1_block}x{k1_block}x{k0_block_max}_r{m0_warp}x{n0_warp}x{k0_warp}_{m1_warp}x{n1_warp}x{k1_warp}_w{m0_warp}x{n0_warp_tile}x{k0_warp_tile}x{m1_warp_tile}x{n1_warp_tile}x{k1_warp_tile}",
        fmt::arg("m0_block", m0_block_),
        fmt::arg("n0_block", n0_block_),
        fmt::arg("k0_block", k0_block_),
        fmt::arg("n1_block", n1_block_),
        fmt::arg("k1_block", k1_block_),
        fmt::arg("k0_block_max", k0_block_max_),
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

std::string FmhaFwdSplitKVTileDesc::Emit() const
{
    std::string       tpl = R"(
    ck_tile::TileFmhaShape<ck_tile::sequence<{{m0_block}}, {{n0_block}}, {{k0_block}}, {{n1_block}}, {{k1_block}}, {{k0_block_max}}>,
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
                                {"k0_block_max", k0_block_max_},
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


std::string FmhaFwdSplitKVCodeGen::GetPadName() const
{
    return Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}",
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
    return Sprintf("fmha_fwd_splitkv_{dtype}_{mode}_{tile_desc}_{pipeline}",
                   fmt::arg("dtype", DataTypeToString(dtype_)),
                   fmt::arg("mode", GetFmhaModeName(mode_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline", GetPipelineConfigName()));
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
        {"mode", problem_.mode_ == FmhaMode::Batch ? "false" : "true"},
        {"mask",
         problem_.mask_type_ == GenericAttentionMaskEnum::NO_MASK || (problem_.window_size_[0] == -1 && problem_.window_size_[1] == -1) ? "false" :
                                                                                                               "true"},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
        {"is_pad_v_head_dim", is_pad_v_head_dim_},
        {"attention_bias", GetBiasClassTag(problem_.bias_enum_)},
        {"is_store_lse", problem_.is_store_lse_},
        {"is_static_quant", problem_.is_static_quant_},
        {"is_paged_kv", problem_.is_paged_kv_},
        {"has_uneven_splits", problem_.has_uneven_splits_},
        {"is_merge_num_head_groups_seq_len_q", problem_.is_merge_num_head_groups_seq_len_q_},
        {"block_per_cu", std::to_string(block_per_cu_)},
        {"pipeline", GetFwdSplitKVPipelineClassTag(pipeline_)}};

    return TEMPLATE_CHECK(source, value_map, "FmhaFwdSplitKVCodeGen::Emit");
}

}  // namespace flashck