#include "core/profiling/moe/moe_sorting_ex/moe_sorting_ex_codegen.h"

namespace flashck {

std::string MoeSortingExCodeGen::GetInstanceName()
{
    auto trait = Sprintf("{sub_token_tile}_{sub_token_one_shot}_{local_token_expert_masking}_{local_token}"
                         "_{skip_expert_with_zero_token}_{expert_tile}",
                    fmt::arg("sub_token_tile", sub_token_tile_),
                    fmt::arg("sub_token_one_shot", sub_token_one_shot_),
                    fmt::arg("local_token_expert_masking", local_token_expert_masking_),
                    fmt::arg("local_token", local_token_),
                    fmt::arg("skip_expert_with_zero_token", skip_expert_with_zero_token_),
                    fmt::arg("expert_tile", expert_tile_));

    auto launch = Sprintf("{max_thread_per_block}_{min_block_per_cu}",
                    fmt::arg("max_thread_per_block", max_thread_per_block_),
                    fmt::arg("min_block_per_cu", min_block_per_cu_));

    return Sprintf("moe_sorting_{problem_name}_{trait}_{launch}",
                   fmt::arg("problem_name", problem_.GetName()),
                   fmt::arg("trait", trait),
                   fmt::arg("launch", launch)
                );
}

std::string MoeSortingExCodeGen::Emit()
{
    std::string tpl = R"(
    using problem_{{idx}} = ck_tile::MoeSortingProblem<index_dtype, 
                                                       weight_dtype, 
                                                       {{sub_token_tile}}, 
                                                       {{sub_token_one_shot}}, 
                                                       {{local_expert_masking}},
                                                       {{local_token}},
                                                       >;
    using {{name}} = ck_tile::MoeSortingKernel<problem_{{idx}}>;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"name", GetInstanceName()},
                                {"idx", idx++},
                                {"sub_token_tile", sub_token_tile_},
                                {"sub_token_one_shot", sub_token_one_shot_},
                                {"local_token_expert_masking", local_token_expert_masking_},
                                {"local_token", local_token_}};

    return TEMPLATE_CHECK(tpl, value_map, "MoeSortingExCodeGen::Emit");

}

}  // namespace flashck