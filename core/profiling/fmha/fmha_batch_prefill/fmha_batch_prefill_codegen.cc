#include "core/profiling/fmha/fmha_batch_prefill/fmha_batch_prefill_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string FmhaBatchPrefillTileDesc::GetInstanceName() const
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

std::string FmhaBatchPrefillTileDesc::Emit() const
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

    return TEMPLATE_CHECK(tpl, value_map, "FmhaBatchPrefillTileDesc::Emit");
}

std::string FmhaBatchPrefillCodeGen::GetPadName() const
{
    // Generate compact padding configuration identifier
    return Sprintf("{q_pad}{kv_pad}{qk_head_pad}{v_head_pad}",
                   fmt::arg("q_pad", is_pad_q_seq_len_ ? "s" : ""),
                   fmt::arg("kv_pad", is_pad_kv_seq_len_ ? "sk" : ""),
                   fmt::arg("qk_head_pad", is_pad_qk_head_dim_ ? "d" : ""),
                   fmt::arg("v_head_pad", is_pad_v_head_dim_ ? "dv" : ""));
}

std::string FmhaBatchPrefillCodeGen::GetPipelineConfigName() const
{
    // Generate complete pipeline configuration identifier
    return Sprintf("{pad_name}_{bias_name}_{quant_mode}{occupancy_override}",
                   fmt::arg("pad_name", GetPadName()),
                   fmt::arg("bias_name", GetBiasShortName(problem_.bias_enum_)),
                   fmt::arg("quant_mode", problem_.is_static_quant_ ? "squant" : "nosquant"),
                   fmt::arg("occupancy_override", 
                           min_block_per_cu_ == -1 ? "" : "_bpc" + std::to_string(min_block_per_cu_)));
}

std::string FmhaBatchPrefillCodeGen::GetInstanceName() const
{
    // Generate unique instance identifier combining all configuration aspects
    return Sprintf("fmha_batch_prefill_{dtype}_{mode}_{tile_desc}_{pipeline_config}",
                   fmt::arg("dtype", DataTypeToString(problem_.dtype_)),
                   fmt::arg("mode", GetFmhaModeName(problem_.mode_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline_config", GetPipelineConfigName()));
}

std::string FmhaBatchPrefillCodeGen::Emit() const
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
    using {{name}}_{{idx}} = ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel<
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
        {"mode", problem_.mode_ == FmhaMode::Batch ? false : true},
        {"is_pad_q_seq_len", is_pad_q_seq_len_},
        {"is_pad_kv_seq_len", is_pad_kv_seq_len_},
        {"is_pad_qk_head_dim", is_pad_qk_head_dim_},
        {"is_pad_v_head_dim", is_pad_v_head_dim_},
        {"has_logits_soft_cap", problem_.has_logits_soft_cap_},
        {"attention_bias", GetBiasClassTag(problem_.bias_enum_)},
        {"is_store_lse", problem_.is_store_lse_},
        {"skip_min_q_seq_len", problem_.is_skip_min_q_seqlen_},
        {"is_static_quant", problem_.is_static_quant_},
        {"block_per_cu", std::to_string(min_block_per_cu_)},
        {"pipeline", GetFwdPipelineClassTag(pipeline_)}
    };

    return TEMPLATE_CHECK(tpl, value_map, "FmhaBatchPrefillCodeGen::Emit");
}

}  // namespace flashck