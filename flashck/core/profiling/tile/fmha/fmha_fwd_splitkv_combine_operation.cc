#include "flashck/core/profiling/fmha_fwd_splitkv_combine_operation.h"

#include "flashck/core/utils/jinjia2_utils.h"

namespace flashck {

std::string FmhaSplitKVCompbineTileDesc::GetInstanceName()
{
    return "b" + std::to_string(bn1_);
};

std::string FmhaFwdSplitKVCombineOperation::GetPipelineConfigName()
{
    std::ostringstream oss;
    oss << (is_static_quant_ ? "squant" : "nosquant")
        << (block_per_cu_ == -1 ? "" : "_" + std::to_string(block_per_cu_));

    return oss.str();
}

std::string FmhaFwdSplitKVCombineOperation::GetInstanceName()
{
    std::ostringstream oss;
    oss << "fmha_fwd_splitkv_combine_d" << hdim_ << "_" << DataTypeToShortString(dtype_) << "_"
        << g_fmha_operation_mode_name_map.at(operation_mode_) << "_" << tile_desc_.GetInstanceName() << "_"
        << GetPipelineConfigName();

    return oss.str();
}

std::string FmhaFwdSplitKVCombineOperation::Emit()
{
    std::string source = R"(
using fmha_fwd_splitkv_combine_pipeline_problem_{{idx}} = ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
    LSEDataType,
    OaccDataType,
    ODataType,
    {{hdim}}, // headdim_v
    {{mode}},
    {{bn1}},
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
                                   {"idx", idx++},
                                   {"hdim", hdim_},
                                   {"bn1", tile_desc_.bn1_},
                                   {"mode", operation_mode_ == FmhaOperationMode::Batch ? "false" : "true"},
                                   {"is_pad_q_seq_len", is_pad_q_seq_len_},
                                   {"is_pad_v_head_dim", is_pad_v_head_dim_},
                                   {"log_max_splits", log_max_splits_},
                                   {"is_static_quant", is_static_quant_},
                                   {"block_per_cu", block_per_cu_}};

    return TemplateLoadAndRender(source, value_map);
}

}  // namespace flashck