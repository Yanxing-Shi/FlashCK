#include "core/profiling/attention/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_codegen.h"

namespace flashck {

std::string FmhaFwdSplitKVCombineTileDesc::GetInstanceName() 
{
    return "b" + std::to_string(n1_block_);
}


std::string FmhaFwdSplitKVCombineCodeGen::GetInstanceName() 
{
    return Sprintf("fmha_fwd_split_kv_combine_{problem_name}_{tile_desc}_{padding}_{min_block_per_cu}",
                   fmt::arg("problem_name", problem_.GetName()),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("padding", GetPaddingConfigName()),
                   fmt::arg("min_block_per_cu", min_block_per_cu_));
}

std::string FmhaFwdSplitKVCombineCodeGen::Emit() 
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

    auto get_log_max_splits_func = [](int num_splits) {
        FC_ENFORCE_GT(num_splits, 0, Unavailable("num splits must > 0"));
        int log_max_splits = 0;
        int val = 1;
        while (val < num_splits) {
            val <<= 1;
            ++log_max_splits;
        }
        return log_max_splits;
    };

    static int idx = 0;

    jinja2::ValuesMap value_map = {{"name", GetInstanceName()},
                                   {"idx", idx++},
                                   {"hdim", problem_.qk_head_dim_},
                                   {"n1_block", tile_desc_.n1_block_},
                                   {"mode", problem_.mode_ == FmhaMode::Batch ? "false" : "true"},
                                   {"is_pad_q_seq_len", is_pad_q_seq_len_},
                                   {"is_pad_v_head_dim", is_pad_v_head_dim_},
                                   {"log_max_splits", get_log_max_splits_func(problem_.num_splits_)},
                                   {"is_static_quant", problem_.is_static_quant_},
                                   {"block_per_cu", min_block_per_cu_}};

    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdSplitKVCombineCodeGen::Emit");
}

}  // namespace flashck