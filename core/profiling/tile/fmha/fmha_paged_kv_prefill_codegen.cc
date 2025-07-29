#include "core/profiling/tile/fmha/fmha_paged_kv_prefill_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string FmhaPagedKVPrefillTileDesc::GetInstanceName() const
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

std::string FmhaPagedKVPrefillTileDesc::Emit() const
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

std::string FmhaPagedKVPrefillCodeGen::GetPadName() const
{
    return Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "sk" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""));
}

std::string FmhaPagedKVPrefillCodeGen::GetPipelineConfigName() const
{
    return Sprintf("{pad_name}_{bias_short_name}_{is_static_quant}{block_per_cu}",
                   fmt::arg("pad_name", GetPadName()),
                   fmt::arg("bias_short_name", GetBiasShortName(bias_enum_)),
                   fmt::arg("is_static_quant", is_static_quant_ ? "squant" : "nosquant"),
                   fmt::arg("block_per_cu", block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_)));
}

std::string FmhaPagedKVPrefillCodeGen::GetInstanceName() const
{
    return Sprintf("fmha_fwd_{dtype}_{mode}_{tile_desc}_{pipeline}",
                   fmt::arg("dtype", DataTypeToString(dtype_)),
                   fmt::arg("mode", GetFmhaModeName(mode_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline", GetPipelineConfigName()));
}

std::string FmhaPagedKVPrefillCodeGen::Emit() const
{
    std::string tpl = R"(
using fmha_pipeline_problem_{{idx}} = ck_tile::BlockFmhaPipelineProblem<
    QDataType,
    KDataType,
    VDataType,
    SaccDataType,
    SMPLComputeDataType,
    BiasDataType,
    RandValOutputDataType,
    LSEDataType,
    PDataType,
    OaccDataType,
    ODataType,
    {{shape}},
    {{mode}},
    ck_tile::ComposedAttention<{{has_logits_soft_cap}} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>,
    ck_tile::TileFmhaTraits<{{is_pad_q_seq_len}},
                            {{is_pad_kv_seq_len}},
                            {{is_pad_qk_head_dim}},
                            {{is_pad_v_head_dim}},
                            {{has_logits_soft_cap}},
                            {{attention_bias}},
                            false, /* kHasBiasGrad */
                            false, /* kStoreLSE_ */
                            false, /* kHasDropout_ */
                            {{is_static_quant}}, /* kDoFp8StaticQuant */
                            {{block_per_cu}}, /* overwrite occupancy if not -1 */
                            {{skip_min_q_seq_len}}, /* skip min seqlen q while chunked prefill */>>;

using {{name}}  =
    ck_tile::FmhaFwdPagedKVKernel<{{pipeline}}<fmha_pipeline_problem_{{idx}}>,
                            ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<OaccDataType, ODataType,
                                                    {{is_pad_q_seq_len}}, {{is_pad_v_head_dim}}>>>;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map = {
        {"name", GetInstanceName()},
        {"idx", idx++},
        {"shape", tile_desc_.Emit()},
        {"mode", problem_.mode_ == FmhaMode::Batch ? false : true},
        {"mask",
         problem_.mask_type_ == GenericAttentionMaskEnum::NO_MASK || (problem_.window_size_[0] == -1 && problem_.window_size_[1] == -1) ? false :
                                                                                                               true},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
        {"is_pad_v_head_dim", is_pad_v_head_dim_},
        {"has_logits_soft_cap", problem_.has_logits_soft_cap_},
        {"attention_bias", GetBiasClassTag(problem_.bias_enum_)},
        {"skip_min_q_seq_len", problem_.is_skip_min_q_seqlen_},
        {"is_static_quant", problem_.is_static_quant_},
        {"block_per_cu", std::to_string(block_per_cu_)},
        {"pipeline", GetFwdPipelineClassTag(pipeline_)}};

    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdCodeGen::Emit");
}

}  // namespace flashck