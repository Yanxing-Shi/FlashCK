#include "core/profiling/gemm/gemm_codegen.h"

namespace flashck {


std::string GemmTileDesc::GetInstanceName() const
{
    // Generate comprehensive tile descriptor name encoding all tiling parameters
    return Sprintf("{m_block}x{n_block}x{k_block}_{m_warp}x{n_warp}x{k_warp}_{m_warp_tile}x{n_warp_tile}x{k_warp_tile}_{a_permute}{b_permute}",
                   fmt::arg("m_block", m_block_),
                   fmt::arg("n_block", n_block_),
                   fmt::arg("k_block", k_block_),
                   fmt::arg("m_warp", m_warp_),
                   fmt::arg("n_warp", n_warp_),
                   fmt::arg("k_warp", k_warp_),
                   fmt::arg("m_warp_tile", m_warp_tile_),
                   fmt::arg("n_warp_tile", n_warp_tile_),
                   fmt::arg("k_warp_tile", k_warp_tile_),
                   fmt::arg("a_permute", a_permute_ ? "ap" : ""),
                   fmt::arg("b_permute", b_permute_ ? "bp" : "")                
                );
}

std::string GemmTileDesc::Emit() const
{
    std::string tile_desc = R"(
        ck_block::TileGemmShape<ck_block::sequence<{{m_block}}, {{n_block}}, {{k_block}}>},
                               ck_block::sequence<{{m_warp}}, {{n_warp}}, {{k_warp}}},
                               ck_block::sequence<{{m_warp_tile}}, {{n_warp_tile}}, {{k_warp_tile}}>,
                               {{a_permute}},
                               {{b_permute}}>,
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
        {"k_warp_tile", k_warp_tile_},
        {"a_permute", a_permute_},
        {"b_permute", b_permute_}
    };

    return TEMPLATE_CHECK(tile_desc, tile_desc_value_map, "GemmTileDesc::Emit");
}

std::string GemmCodeGen::GetInstanceName() const
{
    return Sprintf("{kind_name}_{elementwise_kind}_{a_dtype}_{b_dtype}_{c_dtype}_"
                   "{tile_desc}_{pipeline}_{epilogue}_{scheduler}_{min_block_per_cu}_"
                   "{num_wave_groups}_{tile_partitioner_group_num}_{tile_partitioner_m01}",
                   fmt::arg("kind_name", GetGemmKindShortName(problem_.kind_)),
                   fmt::arg("elementwise_kind", GetElementwiseKindShortName(problem_.elementwise_kind_)),
                   fmt::arg("a_dtype", DataTypeToString(problem_.a_dtype_)),
                   fmt::arg("b_dtype", DataTypeToString(problem_.b_dtype_)),
                   fmt::arg("c_dtype", DataTypeToString(problem_.c_dtype_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("pipeline", GetPipelineVersionEnumShortName(pipeline_version_)),
                   fmt::arg("epilogue", GetEpilogueEnumShortName(pipeline_epilogue_)),
                   fmt::arg("scheduler", GetPipelineSchedulerEnumShortName(pipeline_scheduler_)),
                   fmt::arg("min_block_per_cu", min_block_per_cu_),
                   fmt::arg("num_wave_groups", num_wave_groups_),
                   fmt::arg("tile_partitioner_group_num", tile_partitioner_group_num_),
                   fmt::arg("tile_partitioner_m01", tile_partitioner_m01_)
                );
}

std::string GemmCodeGen::Emit() const
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

using Kernel_{{idx}} = {{kernel}}<TilePartitioner_{{idx}}, GemmPipeline_{{idx}}, GemmEpilogue_{{idx}}>;

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
                                {"use_double_smem_buffer", pipeline_version_ == PipelineVersionEnum::Compute_V4? true:false},
                                {"c_permute", problem_.c_permute_},
                                {"use_structured_sparsity", problem_.use_structured_sparsity_},
                                {"persistent", problem_.is_persistent_},
                                {"preshuffle", problem_.is_preshuffle_},
                                {"kernel", GetGemmKindTag(problem_.kind_)},
                                {"base_pipeline", GetPipelineVersionEnumBaseTag(pipeline_version_)},
                                {"main_pipeline", GetPipelineVersionEnumMainTag(pipeline_version_)},
                                {"epilogue", GetEpilogueEnumTag(pipeline_epilogue_)},
                                {"scheduler", GetPipelineSchedulerEnumTag(pipeline_scheduler_)},
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