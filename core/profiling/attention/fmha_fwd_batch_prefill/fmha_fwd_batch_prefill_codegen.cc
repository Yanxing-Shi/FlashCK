#include "core/profiling/attention/fmha_fwd_batch_prefill/fmha_fwd_batch_prefill_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string FmhaFwdBatchPrefillTileDesc::GetInstanceName()
{
    // Generate comprehensive tile descriptor name encoding all tiling parameters
    return Sprintf(
        "{m0_block}x{n0_block}x{k0_block}_{n1_block}x{k1_block}x{k0_max_block}_"
        "{m0_warp}x{n0_warp}x{k0_warp}_{m1_warp}x{n1_warp}x{k1_warp}_"
        "{m0_warp_tile}x{n0_warp_tile}x{k0_warp_tile}_{m1_warp_tile}x{n1_warp_tile}x{k1_warp_tile}",
        fmt::arg("m0_block", m0_block_),
        fmt::arg("n0_block", n0_block_),
        fmt::arg("k0_block", k0_block_),
        fmt::arg("n1_block", n1_block_),
        fmt::arg("k1_block", k1_block_),
        fmt::arg("k0_max_block", k0_max_block_),
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

std::string FmhaFwdBatchPrefillTileDesc::Emit() 
{
    // Generate TileFmhaShape template instantiation
    std::string tpl = R"(
    ck_tile::TileFmhaShape<
        ck_tile::sequence<{{m0_block}}, {{n0_block}}, {{k0_block}}, {{n1_block}}, {{k1_block}}, {{k0_max_block}}>,
        ck_tile::sequence<{{m0_warp}}, {{n0_warp}}, {{k0_warp}}>,
        ck_tile::sequence<{{m0_warp_tile}}, {{n0_warp_tile}}, {{k0_warp_tile}}>,
        ck_tile::sequence<{{m1_warp}}, {{n1_warp}}, {{k1_warp}}>,
        ck_tile::sequence<{{m1_warp_tile}}, {{n1_warp_tile}}, {{k1_warp_tile}}>,
        true /* kIsVLayoutRowMajor */
    >
)";

    jinja2::ValuesMap value_map{
        {"m0_block", m0_block_},
        {"n0_block", n0_block_},
        {"k0_block", k0_block_},
        {"n1_block", n1_block_},
        {"k1_block", k1_block_},
        {"k0_max_block", k0_max_block_},
        {"m0_warp", m0_warp_},
        {"n0_warp", n0_warp_},
        {"k0_warp", k0_warp_},
        {"m1_warp", m1_warp_},
        {"n1_warp", n1_warp_},
        {"k1_warp", k1_warp_},
        {"m0_warp_tile", m0_warp_tile_},
        {"n0_warp_tile", n0_warp_tile_},
        {"k0_warp_tile", k0_warp_tile_},
        {"m1_warp_tile", m1_warp_tile_},
        {"n1_warp_tile", n1_warp_tile_},
        {"k1_warp_tile", k1_warp_tile_}
    };

    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdBatchPrefillTileDesc::Emit");
}

std::string FmhaFwdBatchPrefillCodeGen::GetInstanceName() 
{
    auto trait = Sprintf("{is_pad_q_seq_len}{is_pad_kv_seq_len}{is_pad_qk_head_dim}{is_pad_v_head_dim}_{is_skip_min_q_seqlen}",
                   fmt::arg("is_pad_q_seq_len", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("is_pad_kv_seq_len", is_pad_kv_seq_len_ ? "sk" : ""),
                   fmt::arg("is_pad_qk_head_dim", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("is_pad_v_head_dim", is_pad_v_head_dim_ ? "dv" : ""),
                   fmt::arg("is_skip_min_q_seqlen", is_skip_min_q_seq_len_ ? "s" : ""));
    
    auto launch = Sprintf("{max_thread_per_block}_{min_block_per_cu}",
                    fmt::arg("max_thread_per_block", max_thread_per_block_),
                    fmt::arg("min_block_per_cu", min_block_per_cu_));

    return Sprintf("fmha_fwd_batch_prefill_{problem_name}_{tile_shape}_{trait}_{strategy}_{launch}",
                   fmt::arg("problem_name", problem_.GetName()),
                   fmt::arg("tile_shape", tile_desc_.GetInstanceName()),
                   fmt::arg("trait", trait),
                   fmt::arg("strategy", GetFwdPipelineShortName(pipeline_)),
                   fmt::arg("launch", launch));
}

std::string FmhaFwdBatchPrefillCodeGen::Emit() 
{
    // Generate complete kernel instantiation template
    std::string tpl = R"(
    // FMHA Batch Prefill Pipeline Problem Configuration
    using fmha_pipeline_problem_{{idx}} = ck_tile::BlockFmhaPipelineProblem<
        QDataType,          // Query data type
        KDataType,          // Key data type  
        VDataType,          // Value data type
        SaccDataType,       // Softmax accumulator type
        SMPLComputeDataType,// Softmax compute type
        BiasDataType,       // Attention bias type
        RandValOutputDataType, // Random value output type
        LSEDataType,        // Log-sum-exp type
        PDataType,          // Probability type
        OaccDataType,       // Output accumulator type
        ODataType,          // Output data type
        {{shape}},          // Tile shape configuration
        {{mode}},           // FMHA mode (batch/grouped)
        ck_tile::ComposedAttention<{{has_logits_soft_cap}} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>,
        ck_tile::TileFmhaTraits<
            {{is_pad_q_seq_len}},    // Query sequence padding
            {{is_pad_kv_seq_len}},   // Key-value sequence padding
            {{is_pad_qk_head_dim}},  // Query-key head dimension padding
            {{is_pad_v_head_dim}},   // Value head dimension padding
            {{has_logits_soft_cap}}, // Logits soft capping
            {{attention_bias}},      // Attention bias type
            false,                   // Bias gradient (not used in inference)
            {{is_store_lse}},        // Store log-sum-exp values
            false,                   // Dropout (not used in inference)
            {{is_static_quant}},     // FP8 static quantization
            {{block_per_cu}},        // Occupancy override (-1 for default)
            {{skip_min_q_seq_len}}   // Skip minimum query length check
        >
    >;

    // Complete Kernel Type Definition
    using {{name}} = ck_tile::FmhaFwdBatchPrefillWithPagedKVCacheKernel<
        {{pipeline}}<fmha_pipeline_problem_{{idx}}>,
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            OaccDataType, 
            ODataType,
            {{is_pad_q_seq_len}}, 
            {{is_pad_v_head_dim}}
        >>
    >;

)";

    static int idx = 0; // Unique index for each generated kernel

    jinja2::ValuesMap value_map = {
        {"name", GetInstanceName()},
        {"idx", idx++},
        {"shape", tile_desc_.Emit()},
        {"mode", false},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
        {"is_pad_v_head_dim", is_pad_v_head_dim_},
        {"has_logits_soft_cap", problem_.has_logits_soft_cap_},
        {"attention_bias", GetBiasClassTag(problem_.bias_enum_)},
        {"is_store_lse", false},
        {"skip_min_q_seq_len", is_skip_min_q_seq_len_},
        {"is_static_quant", problem_.is_static_quant_},
        {"block_per_cu", min_block_per_cu_},
        {"pipeline", GetFwdPipelineClassTag(pipeline_)}
    };

    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdBatchPrefillCodeGen::Emit");
}

}  // namespace flashck