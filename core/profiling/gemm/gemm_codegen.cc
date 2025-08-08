#include "core/profiling/gemm/gemm_codegen.h"

namespace flashck {


std::string GemmTileDesc::GetInstanceName()
{
    // Generate comprehensive tile descriptor name encoding all tiling parameters
    return Sprintf("{m_block}x{n_block}x{k_block}_{m_warp}x{n_warp}x{k_warp}_{m_warp_tile}x{n_warp_tile}x{k_warp_tile}",
                   fmt::arg("m_block", m_block_),
                   fmt::arg("n_block", n_block_),
                   fmt::arg("k_block", k_block_),
                   fmt::arg("m_warp", m_warp_),
                   fmt::arg("n_warp", n_warp_),
                   fmt::arg("k_warp", k_warp_),
                   fmt::arg("m_warp_tile", m_warp_tile_),
                   fmt::arg("n_warp_tile", n_warp_tile_),
                   fmt::arg("k_warp_tile", k_warp_tile_)             
                );
}

std::string GemmTileDesc::Emit()
{
    std::string tile_desc = R"(
        ck_block::TileGemmShape<ck_block::sequence<{{m_block}}, {{n_block}}, {{k_block}}>},
                               ck_block::sequence<{{m_warp}}, {{n_warp}}, {{k_warp}}},
                               ck_block::sequence<{{m_warp_tile}}, {{n_warp_tile}}, {{k_warp_tile}}>>,
)";

    jinja2::ValuesMap tile_desc_value_map = {
        {"m_block", m_block_},
        {"n_block", n_block_},
        {"k_block", k_block_},
        {"m_warp", m_warp_},
        {"n_warp", n_warp_},
        {"k_warp", k_warp_},
        {"m_warp_tile", m_warp_tile_},
        {"n_warp_tile", n_warp_tile_},
        {"k_warp_tile", k_warp_tile_}
    };

    return TEMPLATE_CHECK(tile_desc, tile_desc_value_map, "GemmTileDesc::Emit");
}

std::string GemmCodeGen::GetInstanceName()
{
    auto trait = Sprintf("{is_pad_m}{is_pad_n}{is_pad_k}",
                   fmt::arg("is_pad_m", is_pad_m_ ? "m" : ""),
                   fmt::arg("is_pad_n", is_pad_n_ ? "n" : ""),
                   fmt::arg("is_pad_k", is_pad_k_ ? "k" : ""));

    auto strategy = Sprintf("{pipeline}_{epilogue}_{scheduler}",
                   fmt::arg("pipeline", GetPipelineEnumShortName(pipeline_)),
                   fmt::arg("epilogue", GetEpilogueEnumShortName(epilogue_)),
                   fmt::arg("scheduler", GetSchedulerEnumShortName(scheduler_)));

    auto partition = Sprintf("{tile_partitioner_group_num}_{tile_partitioner_m01}",
                   fmt::arg("tile_partitioner_group_num", tile_partitioner_group_num_),
                   fmt::arg("tile_partitioner_m01", tile_partitioner_m01_));
    
    auto launch = Sprintf("{min_block_per_cu}_{num_wave_groups}",
                   fmt::arg("min_block_per_cu", min_block_per_cu_),
                   fmt::arg("num_wave_groups", num_wave_groups_));

    return Sprintf("{problem}_{trait}_{strategy}_{partition}_{launch}",
                   fmt::arg("problem", problem_.GetName()),
                   fmt::arg("trait", trait),
                   fmt::arg("strategy", strategy),
                    fmt::arg("partition", partition),
                    fmt::arg("launch", launch)
                );
}

std::string GemmCodeGen::Emit()
{
    std::string tpl = R"(
    using TilePartitioner_{{idx}} =
            ck_block::GemmSpatiallyLocalTilePartitioner<{{shape}},
                                                    {{tile_partitioner_group_num}},
                                                    {{tile_partitioner_m01}}>;

    using Traits_{{idx}} = ck_block::TileGemmTraits<{{is_pad_m}},
                                        {{is_pad_n}},
                                        {{is_pad_k}},
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        {{num_wave_groups}}>;

    using GemmUniversalTraits_{{idx}} = ck_block::TileGemmUniversalTraits<{{is_pad_m}},
                            {{is_pad_n}},
                            {{is_pad_k}},
                            {{use_double_smem_buffer}}
                            ALayout,
                            BLayout,
                            CLayout,
                            {{c_permute}},
                            {{use_structured_sparsity}},
                            {{persistent}},
                            {{num_wave_groups}},
                            {{preshuffle}};

    using GemmPipelineProblem_{{idx}} =
        ck_block::GemmPipelineProblem<ADataType, BDataType, AccDataType, {{shape}}, Traits_{{idx}}>;

    using BaseGemmPipeline_{{idx}} = {{base_pipeline}}<GemmPipelineProblem_{{idx}}>;

    const ck_block::index_t k_grain     = args.k_batch * {{block_k}};
    const ck_block::index_t K_split     = (args.K + k_grain - 1) / k_grain * {{block_k}};
    const ck_block::index_t num_loop    = TilePartitioner_{{idx}}::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline_{{idx}}::BlockHasHotloop(num_loop);
    const ck_block::TailNumber tail_num = BaseGemmPipeline_{{idx}}::GetBlockLoopTailNum(num_loop);

    const bool has_hot_loop_v = has_hot_loop.value;
    const auto tail_number_v  = tail_num.value;
    const auto scheduler      = {{scheduler}};
    const auto memory_operation = {{memory_operation}}.value;

    using UniversalGemmProblem_{{idx}} =
                ck_block::UniversalGemmPipelineProblem<ADataType,
                                                        BDataType,
                                                        AccDataType,
                                                        {{shape}},
                                                        GemmUniversalTraits_{{idx}},
                                                        scheduler,
                                                        has_hot_loop_v,
                                                        tail_number_v>;
    using GemmPipeline_{{idx}} = {{main_pipeline}}<UniversalGemmProblem_{{idx}}>;

    {{epilogue}}

    using {{name}} = {{kernel}}<TilePartitioner_{{idx}}, GemmPipeline_{{idx}}, GemmEpilogue_{{idx}}>;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"name", GetInstanceName()},
                                {"idx", idx++},
                                {"shape", tile_desc_.Emit()},
                                {"tile_partitioner_group_num", tile_partitioner_group_num_},
                                {"tile_partitioner_m01", tile_partitioner_m01_},                                
                                {"is_pad_m", is_pad_m_},
                                {"is_pad_n", is_pad_n_},
                                {"is_pad_k", is_pad_k_},
                                {"num_wave_groups", num_wave_groups_},
                                {"use_double_smem_buffer", pipeline_ == PipelineEnum::Compute_V4? true:false},
                                {"c_permute", problem_.c_permute_},
                                {"use_structured_sparsity", use_sparsity_},
                                {"persistent", is_persistent_},
                                {"preshuffle", is_preshuffle_},
                                {"kernel", GetGemmKindTag(problem_.kind_)},
                                {"base_pipeline", GetPipelineEnumBaseTag(pipeline_)},
                                {"main_pipeline", GetPipelineEnumMainTag(pipeline_)},
                                {"epilogue", GetEpilogueEnumTag(epilogue_)},
                                {"scheduler", GetSchedulerEnumTag(scheduler_)},
                                {"m_warp", tile_desc_.m_warp_},
                                {"n_warp", tile_desc_.n_warp_},
                                {"m_warp_tile", tile_desc_.m_warp_tile_},
                                {"n_warp_tile", tile_desc_.n_warp_tile_},
                                {"k_warp_tile", tile_desc_.k_warp_tile_},
                                {"split_k", problem_.split_k_},
                                {"elementwise_kind", GetElementwiseKindTag(problem_.elementwise_kind_)}
};

    return TEMPLATE_CHECK(tpl, value_map, "GemmCodeGen::Emit");
}

}  // namespace flashck