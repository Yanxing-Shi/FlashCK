#include "core/profiling/moe/topk_softmax/topk_softmax_codegen.h"

namespace flashck {

std::string TopKSoftmaxCodeGen::GetInstanceName() const
{
    return Sprintf("topk_softmax_{input_dtype}_{weight_dtype}_{index_dtype}_"
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

std::string TopKSoftmaxCodeGen::Emit() const
{
    std::string tpl = R"(
using ts_problem_{{idx}} = ck_tile::TopkSoftmaxWarpPerRowProblem<{{input_dtype}}, {{weight_dtype}}, {{index_dtype}}, {{num_experts}}>;
using ts_pipeline_{{idx}} = ck_tile::TopkSoftmaxWarpPerRowPipeline<ts_problem_{{idx}}>;
using ts_kernel_{{idx}} = ck_tile::TopkSoftmaxKernel<ts_pipeline_{{idx}}>;

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


    return TEMPLATE_CHECK(tpl, value_map, "TopKSoftmaxCodeGen::Emit");

}

}  // namespace flashck