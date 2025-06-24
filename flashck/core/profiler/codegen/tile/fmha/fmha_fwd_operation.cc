#include "flashck/core/profiler/fmha_fwd_operation.h"

#include "flashck/core/utils/jinjia2_utils.h"

namespace flashck {

std::string FmhaTileDesc::GetConfigName()
{
    return "b" + std::to_string(bm0_) + "x" + std::to_string(bn0_) + "x" + std::to_string(bk0_) + "_"
           + std::to_string(bn1_) + "x" + std::to_string(bk1_) + "x" + std::to_string(bk0_max_) + "_r"
           + std::to_string(rm0_) + "x" + std::to_string(rn0_) + "x" + std::to_string(rk0_) + "_" + std::to_string(rm1_)
           + "x" + std::to_string(rn1_) + "x" + std::to_string(rk1_) + "_w" + std::to_string(wm0_) + "x"
           + std::to_string(wn0_) + "x" + std::to_string(wk0_) + "x" + std::to_string(wm1_) + "x" + std::to_string(wn1_)
           + "x" + std::to_string(wk1_);
};

std::string FmhaTileDesc::Emit()
{
    std::string       source = R"(
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
    return TemplateLoadAndRender(source, value_map);
}

std::string FmhaFwdOperation::GetPadName()
{
    std::ostringstream oss;
    oss << "p" << (is_pad_q_seq_len_ ? "s" : "") << (is_pad_kv_seq_len_ ? "sk" : "") << (is_pad_qk_head_dim_ ? "d" : "")
        << (is_pad_v_head_dim_ ? "dv" : "");
    return oss.str();
}

std::string FmhaFwdOperation::GetPipelineConfigName()
{
    std::ostringstream oss;
    oss << GetPadName() << "_" << g_bias_enum_short_names_map.at(bias_enum_) << "_"
        << (is_static_quant_ ? "squant" : "nosquant")
        << (block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_));
    return oss.str();
}

std::string FmhaFwdOperation::GetConfigName()
{
    std::ostringstream oss;
    oss << "fmha" << "_" << DataTypeToShortString(dtype_) << "_" << g_fmha_operation_mode_name_map.at(operation_mode_)
        << "_" << tile_desc_.GetConfigName() << "_" << GetPipelineConfigName() << "_"
        << g_generic_attention_mask_short_names_map.at(mask_type_);
    return oss.str();
}

std::string FmhaFwdOperation::Emit()
{
    std::string source = R"(
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
        {"is_static_quant", is_static_quant_},
        {"block_per_cu", block_per_cu_},
        {"pipeline", g_block_fmha_fwd_pipeline_map.at(pipeline_)}};

    return TemplateLoadAndRender(source, value_map);
}

}  // namespace flashck