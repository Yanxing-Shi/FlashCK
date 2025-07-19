#include "flashck/core/profiling/tile/fmha/fmha_fwd_appendkv_codegen.h"

#include "flashck/core/utils/macros.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for GEMM operation: 0 - heuristic, 1 - autotuning, 2 - hybrid

namespace flashck {

std::string FmhaAppendKVTileDesc::GetInstanceName() const
{
    return Sprintf("b{bs}x{bsk}x{bd}x{bdv}",
                   fmt::arg("bs", bs_),
                   fmt::arg("bsk", bsk_),
                   fmt::arg("bd", bd_),
                   fmt::arg("bdv", bdv_));
}

std::string FmhaFwdAppendKVCodeGen::GetPadName() const
{
    return Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "sk" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""));
}

std::string FmhaFwdAppendKVCodeGen::GetPipelineConfigName() const
{
    return Sprintf("{pad_name}_{rope_short_name}_{pagedkv}{block_per_cu}",
                   fmt::arg("pad_name", GetPadName()),
                   fmt::arg("rope_short_name", GetRopeShortName(rope_type_)),
                   fmt::arg("pagedkv", is_paged_kv_ ? "pagedkv" : "nopagedkv"),
                   fmt::arg("block_per_cu", block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_)));
}

std::string FmhaFwdAppendKVCodeGen::GetInstanceName() const
{
    return Sprintf("fmha_fwd_appendkv_{dtype}_{mode}_{tile_desc}_{pipeline}",
                   fmt::arg("dtype", DataTypeToString(dtype_)),
                   fmt::arg("mode", GetFmhaModeName(mode_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline", GetPipelineConfigName()));
}

std::string FmhaFwdAppendKVCodeGen::Emit() const
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

    jinja2::ValuesMap value_map = {{"name", GetInstanceName()},
                                   {"idx", std::to_string(idx++)},
                                   {"bs", std::to_string(tile_desc_.bs_)},
                                   {"bsk", std::to_string(tile_desc_.bsk_)},
                                   {"bd", std::to_string(tile_desc_.bd_)},
                                   {"bdv", std::to_string(tile_desc_.bdv_)},
                                   {"rope", GetRopeClassTag(rope_type_)},
                                   {"pagedkv", is_paged_kv_},
                                   {"is_pad_q_seq_len", is_pad_q_seq_len_},
                                   {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
                                   {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
                                   {"is_pad_v_head_dim", is_pad_v_head_dim_},
                                   {"block_per_cu", std::to_string(block_per_cu_)}};

    return TEMPLATE_CHECK(source, value_map, "FmhaFwdAppendKVCodeGen::Emit");
}

}  // namespace flashck