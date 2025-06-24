#include "flashck/core/profiler/fmha_fwd_appendkv_operation.h"

#include "flashck/core/utils/jinjia2_utils.h"

namespace flashck {

std::string FmhaAppendKVTileDesc::GetConfigName()
{
    return "b" + std::to_string(bs_) + "x" + std::to_string(bsk_) + "x" + std::to_string(bd_) + "x"
           + std::to_string(bdv_);
}

std::string FmhaFwdAppendKVOperation::GetPadName()
{
    std::ostringstream oss;
    oss << "p" << (is_pad_q_seq_len_ ? "s" : "") << (is_pad_kv_seq_len_ ? "sk" : "") << (is_pad_qk_head_dim_ ? "d" : "")
        << (is_pad_v_head_dim_ ? "dv" : "");
    return oss.str();
}

std::string FmhaFwdAppendKVOperation::GetPipelineConfigName()
{
    std::ostringstream oss;
    oss << GetPadName() << "_" << g_rope_enum_short_names_map.at(rope_type_) << "_"
        << (is_paged_kv_ ? "pagedkv" : "nopagedkv") << (block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_));
    return oss.str();
}

std::string FmhaFwdAppendKVOperation::GetConfigName()
{
    std::ostringstream oss;
    oss << "fmha_fwd_appendkv" << "_" << DataTypeToShortString(dtype_) << "_"
        << g_fmha_operation_mode_name_map.at(operation_mode_) << "_" << tile_desc_.GetConfigName() << "_"
        << GetPipelineConfigName();
    return oss.str();
}

std::string FmhaFwdAppendKVOperation::Emit()
{
    std::string source = R"(
using fmha_fwd_appendkv_pipeline_problem_{{idx}} = ck_tile::BlockFmhaFwdAppendKVPipelineProblem<
    QDataType,
    KDataType,
    VDataType,
    {{bs}},
    {{bsk}},
    {{bd}},
    {{bdv}},
    true, // kIsVLayoutRowMajor_
    {{rope}},
    {{pagedkv}},
    ck_tile::TileFmhaFwdAppendKVTraits<{{is_pad_q_seq_len}},
                            {{is_pad_kv_seq_len}},
                            {{is_pad_qk_head_dim}},
                            {{is_pad_v_head_dim}},
                            {{block_per_cu}}>>;

using {{name}} =
    ck_tile::FmhaFwdAppendKVKernel<ck_tile::BlockFmhaFwdAppendKVPipeline<fmha_fwd_appendkv_pipeline_problem_{{idx}}>>;

)";
    static int  idx    = 0;

    jinja2::ValuesMap value_map = {{"name", GetConfigName()},
                                   {"idx", idx++},
                                   {"bs", tile_desc_.bs_},
                                   {"bsk", tile_desc_.bsk_},
                                   {"bd", tile_desc_.bd_},
                                   {"bdv", tile_desc_.bdv_},
                                   {"rope", g_rope_enum_tag.at(rope_type_)},
                                   {"pagedkv", is_paged_kv_},
                                   {"is_pad_q_seq_len", is_pad_q_seq_len_},
                                   {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
                                   {"is_pad_qk_head_dim", is_pad_qkv_head_dim_},
                                   {"is_pad_v_head_dim", is_pad_qkv_head_dim_},
                                   {"block_per_cu", block_per_cu_}};

    return TemplateLoadAndRender(source, value_map);
}

}  // namespace flashck