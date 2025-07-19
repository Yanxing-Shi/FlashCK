#include "flashck/core/profiling/tile/fmha/fmha_fwd_codegen.h"

#include "flashck/core/utils/macros.h"

namespace flashck {

std::string FmhaTileDesc::GetInstanceName() const
{
    return Sprintf(
        "b_{bm0}x{bn0}x{bk0}_{bn1}x{bk1}x{bk0_max}_r{rm0}x{rn0}x{rk0}_{rm1}x{rn1}x{rk1}_w{wm0}x{wn0}x{wk0}x{wm1}x{wn1}x{wk1}",
        fmt::arg("bm0", bm0_),
        fmt::arg("bn0", bn0_),
        fmt::arg("bk0", bk0_),
        fmt::arg("bn1", bn1_),
        fmt::arg("bk1", bk1_),
        fmt::arg("bk0_max", bk0_max_),
        fmt::arg("rm0", rm0_),
        fmt::arg("rn0", rn0_),
        fmt::arg("rk0", rk0_),
        fmt::arg("rm1", rm1_),
        fmt::arg("rn1", rn1_),
        fmt::arg("rk1", rk1_),
        fmt::arg("wm0", wm0_),
        fmt::arg("wn0", wn0_),
        fmt::arg("wk0", wk0_),
        fmt::arg("wm1", wm1_),
        fmt::arg("wn1", wn1_),
        fmt::arg("wk1", wk1_));
}

std::string FmhaTileDesc::Emit() const
{
    std::string       tpl = R"(
    ck_tile::TileFmhaShape<ck_tile::sequence<{{bm0}}, {{bn0}}, {{bk0}}, {{bn1}}, {{bk1}}, {{bk0_max}}>,
                            ck_tile::sequence<{{rm0}}, {{rn0}}, {{rk0}}>,
                            ck_tile::sequence<{{wm0}}, {{wn0}}, {{wk0}}>,
                            ck_tile::sequence<{{rm1}}, {{rn1}}, {{rk1}}>,
                            ck_tile::sequence<{{wm1}}, {{wn1}}, {{wk1}}>,
                            true /*kIsVLayoutRowMajor_*/ >
)";
    jinja2::ValuesMap value_map{{"bm0", bm0_},
                                {"bn0", bn0_},
                                {"bk0", bk0_},
                                {"bn1", bn1_},
                                {"bk1", bk1_},
                                {"bk0_max", bk0_max_},
                                {"rm0", rm0_},
                                {"rn0", rn0_},
                                {"rk0", rk0_},
                                {"rm1", rm1_},
                                {"rn1", rn1_},
                                {"rk1", rk1_},
                                {"wm0", wm0_},
                                {"wn0", wn0_},
                                {"wk0", wk0_},
                                {"wm1", wm1_},
                                {"wn1", wn1_},
                                {"wk1", wk1_}};
    return TEMPLATE_CHECK(tpl, value_map, "FmhaTileDesc::Emit");
}

std::string FmhaFwdCodeGen::GetPadName() const
{
    return Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "sk" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""));
}

std::string FmhaFwdCodeGen::GetPipelineConfigName() const
{
    return Sprintf("{pad_name}_{bias_short_name}_{is_static_quant}{block_per_cu}",
                   fmt::arg("pad_name", GetPadName()),
                   fmt::arg("bias_short_name", GetBiasShortName(bias_enum_)),
                   fmt::arg("is_static_quant", is_static_quant_ ? "squant" : "nosquant"),
                   fmt::arg("block_per_cu", block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_)));
}

std::string FmhaFwdCodeGen::GetInstanceName() const
{
    return Sprintf("fmha_fwd_{dtype}_{mode}_{tile_desc}_{pipeline}",
                   fmt::arg("dtype", DataTypeToString(dtype_)),
                   fmt::arg("mode", GetFmhaModeName(mode_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline", GetPipelineConfigName()),
                   fmt::arg("mask", GetAttentionMaskShortName(mask_type_)));
}

std::string FmhaFwdCodeGen::Emit() const
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
    ck_tile::SimplifiedGenericAttentionMask<{{mask}}>,
    ck_tile::TileFmhaTraits<{{is_pad_q_seq_len}},
                            {{is_pad_kv_seq_len}},
                            {{is_pad_qk_head_dim}},
                            {{is_pad_v_head_dim}},
                            {{attention_bias}},
                            false, // kHasBiasGrad
                            false, // kStoreLSE_
                            false, // kHasDropout_
                            {{is_static_quant}}, // kDoFp8StaticQuant
                            {{block_per_cu}}>>;

using {{name}}  =
    ck_tile::FmhaFwdKernel<{{pipeline}}<fmha_pipeline_problem_{{idx}}>,
                            ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<OaccDataType, ODataType,
                                                    {{is_pad_q_seq_len}}, {{is_pad_v_head_dim}}>>>;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map = {
        {"name", GetInstanceName()},
        {"idx", idx++},
        {"shape", tile_desc_.Emit()},
        {"mode", mode_ == FmhaMode::Batch ? false : true},
        {"mask",
         mask_type_ == GenericAttentionMaskEnum::NO_MASK || (window_size_[0] == -1 && window_size_[1] == -1) ? false :
                                                                                                               true},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
        {"is_pad_v_head_dim", is_pad_v_head_dim_},
        {"attention_bias", GetBiasClassTag(bias_enum_)},
        {"is_static_quant", is_static_quant_},
        {"block_per_cu", std::to_string(block_per_cu_)},
        {"pipeline", GetFwdPipelineClassTag(pipeline_)}};

    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdCodeGen::Emit");
}

}  // namespace flashck