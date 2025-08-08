#include "core/profiling/moe/moe_gemm/moe_gemm_codegen.h"

#include "core/utils/macros.h"

namespace flashck {

std::string MoeGemmTileDesc::GetInstanceName()
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

std::string MoeGemmTileDesc::Emit()
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

std::string MoeGemmCodeGen::GetInstanceName()
{
    auto trait = Sprintf("{is_pad_hidden_size}{is_pad_intermediate_size}{is_interleave}",
                   fmt::arg("is_pad_hidden_size", is_pad_hidden_size_ ? "h" : ""),
                   fmt::arg("is_pad_intermediate_size", is_pad_intermediate_size_ ? "i" : ""),
                   fmt::arg("is_interleave", is_interleave_ ? "i" : ""));
    
    auto launch = Sprintf("{max_thread_per_block}_{min_block_per_cu}",
                     fmt::arg("max_thread_per_block", max_thread_per_block_),
                        fmt::arg("min_block_per_cu", min_block_per_cu_));   

    return Sprintf("moe_gemm_{problem}_{tile_shape}_{trait}_{launch}",
                   fmt::arg("problem", problem_.GetName()),
                   fmt::arg("tile_shape", tile_desc_.GetInstanceName()),
                   fmt::arg("trait", trait),
                   fmt::arg("launch", launch));
}

std::string MoeGemmCodeGen::Emit()
{
    // Generate complete MoE GEMM kernel instantiation template
    std::string tpl = R"(
    // MoE GEMM Pipeline Traits Configuration
    using moe_traits_{{idx}} = ck_tile::FusedMoeGemmTraits<
        {{is_only_gate}},        // Gate-only computation flag
        {{use_smooth_quant}},    // Smooth quantization support
        1,     // Atomic operations mode, 0-no atomic, 1-atomic-pk-f16/bf16, 2-atomic-f32
        FusedMoeGemmWeightPermuteEnum::b_nr_kr_waveflatten, // Weight permutation strategy
        {{is_pad_hidden_size}}, // Hidden size padding flag
        {{is_pad_intermediate_size}}, // Intermediate size padding flag
        {{is_interleave}} // Interleave flag
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
    using {{name}} = ck_tile::FusedMoeGemmKernel<
        moe_partitioner_{{idx}}, 
        moe_pipeline_{{idx}}, 
        void                     // Custom epilogue (if needed)
    >; 

)";
    
    static int idx = 0; // Unique index for each generated kernel

    jinja2::ValuesMap value_map{
        {"name", GetInstanceName()},
        {"idx", idx++},
        {"is_pad_hidden_size", is_pad_hidden_size_},
        {"is_pad_intermediate_size", is_pad_intermediate_size_},
        {"is_interleave", is_interleave_},
        {"shape", tile_desc_.Emit()},
        {"activation", GetActivationTag(problem_.activation_)},
        {"is_only_gate", problem_.is_only_gate_},
        {"use_smooth_quant", problem_.use_smooth_quant_}
    };

    return TEMPLATE_CHECK(tpl, value_map, "MoeGemmCodeGen::Emit");
}

}  // namespace flashck