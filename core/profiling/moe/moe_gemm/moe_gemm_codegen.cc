#include "core/profiling/moe/fused_moe_codegen.h"

namespace flashck {

std::string FmhaFwdTileDesc::GetInstanceName() const
{
    return Sprintf(
        "{m0_block}x{n0_block}x{k0_block}_{n1_block}x{k1_block}x{k0_max_block}_r{m0_warp}x{n0_warp}x{k0_warp}_{m1_warp}x{n1_warp}x{k1_warp}_w{m0_warp}x{n0_warp_tile}x{k0_warp_tile}x{m1_warp_tile}x{n1_warp_tile}x{k1_warp_tile}",
        fmt::arg("m0_block", m0_block_),
        fmt::arg("n0_block", n0_block_),
        fmt::arg("k0_block", k0_block_),
        fmt::arg("m0_block", m0_block_),
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

std::string FmhaFwdTileDesc::Emit() const
{
    std::string       tpl = R"(
    ck_tile::FusedMoeGemmShape<ck_tile::sequence<{{m0_block}}, {{n0_block}}, {{k0_block}}, {{n1_block}}, {{k1_block}}, {{k0_max_block}}>,
                            ck_tile::sequence<{{m0_warp}}, {{n0_warp}}, {{k0_warp}}>,
                            ck_tile::sequence<{{m0_warp}}, {{n0_warp_tile}}, {{k0_warp_tile}}>,
                            ck_tile::sequence<{{m1_warp}}, {{n1_warp}}, {{k1_warp}}>,
                            ck_tile::sequence<{{m1_warp_tile}}, {{n1_warp_tile}}, {{k1_warp_tile}}>,
                            true /*kIsVLayoutRowMajor_*/ >
)";
    jinja2::ValuesMap value_map{{"m0_block", m0_block_},
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
                                {"m0_warp", m0_warp_},
                                {"n0_warp_tile", n0_warp_tile_},
                                {"k0_warp_tile", k0_warp_tile_},
                                {"m1_warp_tile", m1_warp_tile_},
                                {"n1_warp_tile", n1_warp_tile_},
                                {"k1_warp_tile", k1_warp_tile_}};
    return TEMPLATE_CHECK(tpl, value_map, "FmhaFwdTileDesc::Emit");
}


std::string MoeGemmCodeGen::GetInstanceName() const
{
    return Sprintf("moe_gemm_{input_dtype}_{weight_dtype}_{index_dtype}_"
                   "{num_experts}_{issue_per_col}_{bytes_per_issue}_{launch_type}_{block_size}_{min_block_per_cu}",
                   fmt::arg("input_dtype", DataTypeToString(problem_.input_dtype_)),
                   fmt::arg("weight_dtype", DataTypeToString(problem_.weight_dtype_)),
                   fmt::arg("index_dtype", DataTypeToString(problem_.index_dtype_)),
                   fmt::arg("num_experts", num_experts_),
                   fmt::arg("issue_per_col", issues_pre_col_),
                   fmt::arg("bytes_per_issue", bytes_per_issue_),
                   fmt::arg("launch_type", launch_type_),
                   fmt::arg("block_size", block_size_),
                   fmt::arg("min_block_per_cu", min_block_per_cu_)
                );
}

std::string MoeGemmCodeGen::Emit() const
{
    std::string tpl = R"(
    
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly, Ts_::FusedQuant == 1, 1 /*atomic*/>;
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
                                                                 {{act}}, // TODO: hardcoded
                                                                f_shape,
                                                                f_traits>;

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
                                {"min_block_per_cu", min_block_per_cu_}
                               };


    return TEMPLATE_CHECK(tpl, value_map, "MoeGemmCodeGen::Emit");

}

}  // namespace flashck