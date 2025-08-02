#include "core/profiling/moe/moe_gemm_codegen.h"

namespace flashck {

#include "core/profiling/moe/moe_gemm/moe_gemm_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string MoeGemmTileDesc::GetInstanceName() const
{
    // Generate comprehensive dual-stage MoE GEMM tile descriptor name
    return Sprintf(
        "{m0_block}x{n0_block}x{k0_block}_{m1_block}x{n1_block}x{k1_block}_{m0_warp}x{n0_warp}x{k0_warp}_{m1_warp}x{n1_warp}x{k1_warp}_wt{m0_warp_tile}x{n0_warp_tile}x{k0_warp_tile}x{m1_warp_tile}x{n1_warp_tile}x{k1_warp_tile}",
        fmt::arg("m0_block", m0_block_),
        fmt::arg("n0_block", n0_block_),
        fmt::arg("k0_block", k0_block_),
        fmt::arg("m1_block", m1_block_),
        fmt::arg("n1_block", n1_block_),
        fmt::arg("k1_block", k1_block_),
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

std::string MoeGemmTileDesc::Emit() const
{
    // Generate FusedMoeGemmShape template instantiation with dual-stage hierarchical tiling
    std::string tpl = R"(
    ck_tile::FusedMoeGemmShape<
        ck_tile::sequence<{{m0_block}}, {{n0_block}}, {{k0_block}}>, /* Stage 0: Token-to-Intermediate */
        ck_tile::sequence<{{m0_warp}}, {{n0_warp}}, {{k0_warp}}>,     /* Stage 0: Warp distribution */
        ck_tile::sequence<{{m0_warp_tile}}, {{n0_warp_tile}}, {{k0_warp_tile}}>, /* Stage 0: Thread tiles */
        ck_tile::sequence<{{m1_block}}, {{n1_block}}, {{k1_block}}>, /* Stage 1: Intermediate-to-Output */
        ck_tile::sequence<{{m1_warp}}, {{n1_warp}}, {{k1_warp}}>,     /* Stage 1: Warp distribution */
        ck_tile::sequence<{{m1_warp_tile}}, {{n1_warp_tile}}, {{k1_warp_tile}}> /* Stage 1: Thread tiles */
    >
)";
    
    jinja2::ValuesMap value_map{
        {"m0_block", m0_block_},
        {"n0_block", n0_block_},
        {"k0_block", k0_block_},
        {"m1_block", m1_block_},
        {"n1_block", n1_block_},
        {"k1_block", k1_block_},
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
        {"k1_warp_tile", k1_warp_tile_}};
    
    return TEMPLATE_CHECK(tpl, value_map, "MoeGemmTileDesc::Emit");
}

std::string MoeGemmCodeGen::GetInstanceName() const
{
    // Generate unique instance identifier combining all MoE GEMM configuration aspects
    return Sprintf("moe_gemm_{input_dtype}_{weight_dtype}_{index_dtype}_{tile}_{num_experts}_{activation}_{launch_config}",
                   fmt::arg("input_dtype", DataTypeToString(problem_.input_dtype_)),
                   fmt::arg("weight_dtype", DataTypeToString(problem_.weight_dtype_)),
                   fmt::arg("index_dtype", DataTypeToString(problem_.index_dtype_)),
                   fmt::arg("tile", tile_desc_.GetInstanceName()),
                   fmt::arg("num_experts", num_experts_),
                   fmt::arg("activation", GetActivationEnumShortName(act_)),
                   fmt::arg("launch_config", Sprintf("ipc{}_bpi{}_lt{}_bs{}_bpc{}",
                           issues_pre_col_, bytes_per_issue_, launch_type_, 
                           block_size_, min_block_per_cu_))
                );
}

std::string MoeGemmCodeGen::Emit() const
{
    // Generate complete MoE GEMM kernel instantiation template
    std::string tpl = R"(
    // MoE GEMM Pipeline Traits Configuration
    using moe_traits_{{idx}} = ck_tile::FusedMoeGemmTraits<
        {{is_only_gate}},        // Gate-only computation flag
        {{use_smooth_quant}},    // Smooth quantization support
        1                        // Atomic operations mode
    >;

    // MoE GEMM Pipeline Problem Definition
    using moe_problem_{{idx}} = ck_tile::FusedMoeGemmPipelineProblem<
        ADataType,               // Input token data type
        GDataType,               // Gate weight data type
        DDataType,               // Down projection weight data type
        AccDataType,             // Accumulator data type
        ODataType,               // Output data type
        AScaleDataType,          // Input scaling data type
        GScaleDataType,          // Gate scaling data type
        DScaleDataType,          // Down scaling data type
        YSmoothScaleDataType,    // Smooth quantization scaling type
        TopkWeightDataType,      // TopK routing weight data type
        IndexDataType,           // Expert index data type
        {{activation}},          // Activation function (SwiGLU, GELU, etc.)
        {{shape}},               // Dual-stage tile shape configuration
        moe_traits_{{idx}}       // MoE-specific traits
    >;

    // MoE GEMM Pipeline Implementation
    using moe_pipeline_{{idx}} = ck_tile::FusedMoeGemmPipeline_FlatmmUk<moe_problem_{{idx}}>;

    // Tile Partitioner for Expert Load Balancing
    using moe_partitioner_{{idx}} = ck_tile::FusedMoeGemmTilePartitioner_Linear<{{shape}}>;

    // Complete MoE GEMM Kernel Definition
    using moe_kernel_{{idx}} = ck_tile::FusedMoeGemmKernel<
        moe_partitioner_{{idx}}, 
        moe_pipeline_{{idx}}, 
        void                     // Custom epilogue (if needed)
    >; 

)";
    
    static int idx = 0; // Unique index for each generated kernel

    jinja2::ValuesMap value_map{
        {"idx", idx++},
        {"input_dtype", DataTypeToString(problem_.input_dtype_)},
        {"weight_dtype", DataTypeToString(problem_.weight_dtype_)},
        {"index_dtype", DataTypeToString(problem_.index_dtype_)},
        {"num_experts", problem_.num_experts_},
        {"issue_per_col", issues_pre_col_},
        {"bytes_per_issue", bytes_per_issue_},
        {"launch_type", launch_type_},
        {"block_size", block_size_},
        {"min_block_per_cu", min_block_per_cu_},
        {"shape", tile_desc_.Emit()},
        {"activation", GetActivationTag(act_)},
        {"is_only_gate", false},  // Configurable based on MoE architecture
        {"use_smooth_quant", problem_.use_smooth_quant_}
    };

    return TEMPLATE_CHECK(tpl, value_map, "MoeGemmCodeGen::Emit");
}

std::string MoeGemmTileDesc::Emit() 
{
    std::string       tpl = R"(
    ck_tile::FusedMoeGemmShape<ck_tile::sequence<{{m0_block}}, {{n0_block}}, {{k0_block}}},
                            ck_tile::sequence<{{m0_warp}}, {{n0_warp}}, {{k0_warp}}>,
                            ck_tile::sequence<{{m0_warp_tile}}, {{n0_warp_tile}}, {{k0_warp_tile}}>,
                            ck_tile::sequence<{{m1_block}}, {{n1_block}}, {{k1_block}}},
                            ck_tile::sequence<{{m1_warp}}, {{n1_warp}}, {{k1_warp}}},
                            ck_tile::sequence<{{m1_warp_tile}}, {{n1_warp_tile}}, {{k1_warp_tile}}>
                             >
)";
    jinja2::ValuesMap value_map{
        {"m0_block", m0_block_},
        {"n0_block", n0_block_},
        {"k0_block", k0_block_},
        {"m1_block", m1_block_},
        {"n1_block", n1_block_},
        {"k1_block", k1_block_},
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
        {"k1_warp_tile", k1_warp_tile_}};
    return TEMPLATE_CHECK(tpl, value_map, "MoeGemmTileDesc::Emit");
}


std::string MoeGemmCodeGen::GetInstanceName() 
{
    return Sprintf("moe_gemm_{input_dtype}_{weight_dtype}_{index_dtype}_"
                   "{tile}_{num_experts}_{issue_per_col}_{bytes_per_issue}_{launch_type}_{block_size}_{min_block_per_cu}",
                   fmt::arg("input_dtype", DataTypeToString(problem_.input_dtype_)),
                   fmt::arg("weight_dtype", DataTypeToString(problem_.weight_dtype_)),
                   fmt::arg("index_dtype", DataTypeToString(problem_.index_dtype_)),
                   fmt::arg("tile", tile_desc_.GetInstanceName()),
                   fmt::arg("num_experts", num_experts_),
                   fmt::arg("issue_per_col", issues_pre_col_),
                   fmt::arg("bytes_per_issue", bytes_per_issue_),
                   fmt::arg("launch_type", launch_type_),
                   fmt::arg("block_size", block_size_),
                   fmt::arg("min_block_per_cu", min_block_per_cu_)
                );
}

std::string MoeGemmCodeGen::Emit() 
{
    std::string tpl = R"(
    using traits_{{idx}} = ck_tile::FusedMoeGemmTraits<{{is_only_gate}}, {{use_smooth_quant}}, 1 /*atomic*/>;
    using problem_{{idx}} = ck_tile::FusedMoeGemmPipelineProblem<ADataType,
                                                                 GDataType,
                                                                 DDataType,
                                                                 AccDataType,
                                                                 ODataType,
                                                                 AScaleDataType,
                                                                 GScaleDataType,
                                                                 DScaleDataType,
                                                                 YSmoothScaleDataType,
                                                                 TopkWeightDataType,
                                                                 IndexDataType,
                                                                 {{activation}}, // TODO: hardcoded
                                                                 {{shape}},
                                                                traits_{{idx}}>;

    using pipeline_{{idx}}    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<problem_{{idx}}>;
    using partitioner_{{idx}} = ck_tile::FusedMoeGemmTilePartitioner_Linear<shape_{{idx}}>;
    using kernel_{{idx}}  = ck_tile::FusedMoeGemmKernel<partitioner_{{idx}}, pipeline_{{idx}}, void>; 
)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"idx", idx++},
                                {"input_dtype", DataTypeToString(problem_.input_dtype_)},
                                {"weight_dtype", DataTypeToString(problem_.weight_dtype_)},
                                {"index_dtype", DataTypeToString(problem_.index_dtype_)},
                                {"num_experts", problem_.num_experts_},
                                {"issue_per_col", issues_pre_col_},
                                {"bytes_per_issue", bytes_per_issue_},
                                {"launch_type", launch_type_},
                                {"block_size", block_size_},
                                {"min_block_per_cu", min_block_per_cu_},
                                {"shape", tile_desc_.Emit()},
                                {"activation", GetActivationTag(act_)}
                               };


    return TEMPLATE_CHECK(tpl, value_map, "MoeGemmCodeGen::Emit");

}

}  // namespace flashck