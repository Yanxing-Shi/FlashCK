#include "core/profiling/moe/moe_sorting/moe_sorting_codegen.h"

namespace flashck {

std::string MoeSortingCodeGen::GetInstanceName() const
{
    return Sprintf("moe_sorting_{weight_dtype}_{index_dtype}_"
                   "{unroll_num}_{expert_tile}_{min_block_per_cu}",
                   fmt::arg("weight_dtype", DataTypeToString(problem_.weight_dtype_)),
                   fmt::arg("index_dtype", DataTypeToString(problem_.index_dtype_)),
                   fmt::arg("unroll_num", internal_load_unroll_),
                   fmt::arg("expert_tile", expert_tile_),
                   fmt::arg("min_block_per_cu", min_block_per_cu_)
                );
}

std::string MoeSortingCodeGen::Emit() const
{
    std::string tpl = R"(
    using ms_problem_{{idx}} = ck_tile::MoeSortingProblem<{{index_dtype}}, {{weight_dtype}}, {{unroll_num}}, {{expert_tile}}>;
    using ms_kernel_{{idx}} = ck_tile::MoeSortingKernel<ms_problem_{{idx}}>;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"idx", idx++},
                                {"weight_dtype", DataTypeToString(problem_.weight_dtype_)},
                                {"index_dtype", DataTypeToString(problem_.index_dtype_)},
                                {"unroll_num", internal_load_unroll_},
                                {"expert_tile", expert_tile_}
                               };


    return TEMPLATE_CHECK(tpl, value_map, "MoeSortingCodeGen::Emit");

}

}  // namespace flashck