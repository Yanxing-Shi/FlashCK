#include "core/profiling/fmha/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_codegen.h"

namespace flashck {

std::string FmhaFwdSplitKVCombineTileDesc::GetInstanceName() const
{
    return "b" + std::to_string(n1_block_);
}

std::string FmhaFwdSplitKVCombineCodeGen::GetPipelineConfigName() const
{
    return Sprintf("{is_static_quant}_{block_per_cu}",
                   fmt::arg("is_static_quant", problem_.is_static_quant_ ? "squant" : "nosquant"),
                   fmt::arg("block_per_cu", min_block_per_cu_ == -1 ? "" : "_" + std::to_string(min_block_per_cu_)));
}

std::string FmhaFwdSplitKVCombineCodeGen::GetInstanceName() const
{
    return Sprintf("fmha_fwd_splitkv_combine_{dtype}_{mode}_{tile_desc}_{pipeline}",
                   fmt::arg("dtype", DataTypeToString(problem_.dtype_)),
                   fmt::arg("mode", GetFmhaModeName(problem_.mode_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline", GetPipelineConfigName()));
}

std::string FmhaFwdSplitKVCombineCodeGen::Emit() const
{
    std::string tpl = R"(
using fmha_fwd_splitkv_combine_pipeline_problem_{{idx}} = ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
    LSEDataType,
    OaccDataType,
    ODataType,
    {{hdim}}, // headdim_v
    {{mode}},
    {{n1_block}},
    ck_tile::TileFmhaFwdSplitKVCombineTraits<{{is_pad_q_seq_len}},
                                            {{is_pad_v_head_dim}},
                                            false, // kStoreLSE_
                                            {{is_static_quant}},
                                            {{log_max_splits}},
                                            {{block_per_cu}}>>;


/// FIXME: use {spad}/{dvpad} as kPadM/kPadN parameters after solving
///        store_tile_raw() data corruption issue
using {{name}} =
    ck_tile::FmhaFwdSplitKVCombineKernel<ck_tile::BlockFmhaFwdSplitKVCombinePipeline<fmha_fwd_splitkv_combine_pipeline_problem_{{idx}}>,
                                        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<OaccDataType,
                                                                  ODataType,
                                                                  false, // kPadM_
                                                                  false // kPadN_
                                                                  >>>;

)";

    static int idx = 0;

    jinja2::ValuesMap value_map = {{"name", GetInstanceName()},
                                   {"idx", std::to_string(idx++)},
                                   {"hdim", std::to_string(problem_.qk_head_dim_)},
                                   {"n1_block", std::to_string(tile_desc_.n1_block_)},
                                   {"mode", problem_.mode_ == FmhaMode::Batch ? "false" : "true"},
                                   {"is_pad_q_seq_len", is_pad_q_seq_len_},
                                   {"is_pad_v_head_dim", is_pad_v_head_dim_},
                                   {"log_max_splits", std::to_string(log_max_splits_)},
                                   {"is_static_quant", problem_.is_static_quant_},
                                   {"block_per_cu", std::to_string(min_block_per_cu_)}};

    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdSplitKVCombineCodeGen::Emit");
}

}  // namespace flashck