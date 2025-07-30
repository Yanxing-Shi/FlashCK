#include "core/profiling/tile/fmha/fmha_fwd_append_kv/fmha_fwd_append_kv_codegen.h"

#include "core/utils/macros.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for GEMM operation: 0 - heuristic, 1 - autotuning, 2 - hybrid

namespace flashck {

std::string FmhaFwdAppendKVTileDesc::GetInstanceName() const
{
    return Sprintf("{s_block}x{sk_block}x{d_block}x{dv_block}",
                   fmt::arg("s_block", s_block_),
                   fmt::arg("sk_block", sk_block_),
                   fmt::arg("d_block", d_block_),
                   fmt::arg("dv_block", dv_block_));
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
                   fmt::arg("rope_short_name", GetRopeShortName(problem_.rope_type_)),
                   fmt::arg("pagedkv", problem_.paged_block_size_ > 0 ? "pagedkv" : "nopagedkv"),
                   fmt::arg("block_per_cu", min_block_per_cu_ == -1 ? "" : "_" + std::to_string(min_block_per_cu_)));
}

std::string FmhaFwdAppendKVCodeGen::GetInstanceName() const
{
    return Sprintf("fmha_fwd_appendkv_{dtype}_{mode}_{tile_desc}_{pipeline}",
                   fmt::arg("dtype", DataTypeToString(problem_.dtype_)),
                   fmt::arg("mode", GetFmhaModeName(problem_.mode_)),
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
    {{s_block}},
    {{sk_block}},
    {{d_block}},
    {{dv_block}},
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
                                   {"s_block", std::to_string(tile_desc_.s_block_)},
                                   {"sk_block", std::to_string(tile_desc_.sk_block_)},
                                   {"d_block", std::to_string(tile_desc_.d_block_)},
                                   {"dv_block", std::to_string(tile_desc_.dv_block_)},
                                   {"rope", GetRopeClassTag(problem_.rope_type_)},
                                   {"pagedkv", problem_.paged_block_size_ > 0 ? true : false},
                                   {"is_pad_q_seq_len", is_pad_q_seq_len_},
                                   {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
                                   {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
                                   {"is_pad_v_head_dim", is_pad_v_head_dim_},
                                   {"block_per_cu", std::to_string(min_block_per_cu_)}};

    return TEMPLATE_CHECK(source, value_map, "FmhaFwdAppendKVCodeGen::Emit");
}

}  // namespace flashck