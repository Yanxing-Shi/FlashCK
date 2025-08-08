#include "core/profiling/attention/fmha_fwd_append_kv/fmha_fwd_append_kv_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string FmhaFwdAppendKVTileDesc::GetInstanceName()
{
    return Sprintf("{s_block}x{sk_block}x{d_block}x{dv_block}",
                   fmt::arg("s_block", s_block_),
                   fmt::arg("sk_block", sk_block_),
                   fmt::arg("d_block", d_block_),
                   fmt::arg("dv_block", dv_block_));
}

std::string FmhaFwdAppendKVCodeGen::GetInstanceName() 
{   
    auto trait = Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "sk" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""));

    auto launch = Sprintf("{max_thread_per_block}_{min_block_per_cu}",
                   fmt::arg("max_thread_per_block", max_thread_per_block_),
                   fmt::arg("min_block_per_cu", min_block_per_cu_));

    return Sprintf("fmha_fwd_appendkv_{problem}_{tile_shape}_{trait}_{launch}",
                   fmt::arg("problem", problem_.GetName()),
                   fmt::arg("tile_shape", tile_desc_.GetInstanceName()),
                   fmt::arg("trait", trait),
                   fmt::arg("launch", launch));
}

std::string FmhaFwdAppendKVCodeGen::Emit() 
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