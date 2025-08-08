#include "core/profiling/moe/moe_sorting/moe_sorting_codegen.h"

namespace flashck {

std::string MoeSortingCodeGen::GetInstanceName()
{
    auto trait = Sprintf("{load_unroll}_{expert_tile}",
                    fmt::arg("load_unroll", load_unroll_),
                    fmt::arg("expert_tile", expert_tile_));

    auto launch = Sprintf("{max_thread_per_block}_{min_block_per_cu}",
                    fmt::arg("max_thread_per_block", max_thread_per_block_),
                    fmt::arg("min_block_per_cu", min_block_per_cu_));

    return Sprintf("moe_sorting_{problem_name}_{trait}_{launch}",
                   fmt::arg("problem_name", problem_.GetName()),
                   fmt::arg("trait", trait),
                   fmt::arg("launch", launch));
}

std::string MoeSortingCodeGen::Emit()
{
    std::string tpl = R"(
    using problem_{{idx}} = ck_tile::MoeSortingProblem<index_dtype, weight_dtype, {{unroll_num}}, {{expert_tile}}>;
    using {{name}} = ck_tile::MoeSortingKernel<problem_{{idx}}>;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"name", GetInstanceName()},
                                {"idx", idx++},
                                {"unroll_num", load_unroll_},
                                {"expert_tile", expert_tile_}};


    return TEMPLATE_CHECK(tpl, value_map, "MoeSortingCodeGen::Emit");

}

}  // namespace flashck