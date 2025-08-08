#include "core/profiling/moe/moe_smooth_quant/moe_smooth_quant_codegen.h"

namespace flashck {

std::string MoeSmoothQuantTileDesc::GetInstanceName()
{
    return Sprintf("{repeat_m}_{repeat_n}_{thread_per_block_m}_{thread_per_block_n}_{vector_n}",
                   fmt::arg("repeat_m", m_repeat_),
                   fmt::arg("repeat_n", n_repeat_),
                   fmt::arg("thread_per_block_m", m_thread_per_block_),
                   fmt::arg("thread_per_block_n", n_thread_per_block_),
                   fmt::arg("vector_n", n_vector_));
}

std::string MoeSmoothQuantTileDesc::Emit() 
{
    bool is_warp_per_row = n_thread_per_block_ <= warpSize;
    FC_ENFORCE_EQ((m_thread_per_block_ * n_thread_per_block_) % warpSize,
                  0,
                  Unavailable("m_thread_per_block_ * n_thread_per_block_ must be multiple of warpSize"));

    int64_t total_warps = (m_thread_per_block_ * n_thread_per_block_) / warpSize;
    // num of warps along m
    int64_t block_warps_m = [&]() -> int64_t {
        if (is_warp_per_row) {
            FC_ENFORCE_EQ(
                warpSize % n_thread_per_block_, 0, Unavailable("n_thread_per_block_ must be multiple of warpSize"));
            return total_warps * (warpSize / n_thread_per_block_);
        }
        else {
            // static_assert(warpSize % m_thread_per_block_ == 0);
            return total_warps / (n_thread_per_block_ / warpSize);
        }
    }();

    // num of warps along n
    int64_t block_warps_n = [&]() -> int64_t {
        if (is_warp_per_row) {
            FC_ENFORCE_EQ(
                warpSize % n_thread_per_block_, 0, Unavailable("n_thread_per_block_ must be multiple of warpSize"));
            return 1;
        }
        else {
            FC_ENFORCE_EQ(
                n_thread_per_block_ % warpSize, 0, Unavailable("n_thread_per_block_ must be multiple of warpSize"));

            return n_thread_per_block_ / warpSize;
        }
    }();

    int64_t block_m = m_repeat_ * m_thread_per_block_;
    int64_t block_n = n_repeat_ * n_thread_per_block_ * n_vector_;

    int64_t warp_m = m_thread_per_block_ / block_warps_m;
    int64_t warp_n = n_thread_per_block_ / block_warps_n * n_vector_;

    std::string tile_desc = R"(
    ck_tile::Generic2dBlockShape<ck_tile::sequence<{{block_m}}, {{block_n}}>,
                                ck_tile::sequence<{{block_warps_m}}, {{block_warps_n}}>,
                                ck_tile::sequence<{{warp_m}}, {{warp_n}}>, 
                                ck_tile::sequence<1, {{vector_n}}>>,
)";

    jinja2::ValuesMap tile_desc_value_map = {
        {"block_m", block_m},
        {"block_n", block_n},
        {"block_warps_m", block_warps_m},
        {"block_warps_n", block_warps_n},
        {"warp_m", warp_m},
        {"warp_n", warp_n},
        {"vector_n", n_vector_},
    };

    return TEMPLATE_CHECK(tile_desc, tile_desc_value_map, "MoeSmoothQuantTileDesc::Emit");
}


std::string MoeSmoothQuantCodeGen::GetInstanceName()
{   
    auto launch = Sprintf("{max_thread_per_block}_{min_block_per_cu}",
                   fmt::arg("max_thread_per_block", max_thread_per_block_),
                   fmt::arg("min_block_per_cu", min_block_per_cu_));

    return Sprintf("moe_smooth_quant_{problem_name}_{trait}_{strategy}_{launch}",
                   fmt::arg("problem_name", problem_.GetName()),
                   fmt::arg("trait", is_pad_n_ ? "pad" : "no_pad"),
                   fmt::arg("strategy", is_two_pass_ ? "two_pass" : "one_pass"),
                   fmt::arg("launch", launch));
}

std::string MoeSmoothQuantCodeGen::Emit()
{
    std::string tpl = R"(
    using pipeline_problem_{{idx}} = ck_tile::SmoothquantPipelineProblem<
        XDataType,
        SmoothScaleDataType,
        ComputeDataType,
        YScaleDataType,
        QYDataType,
        {{shape}},
        {{is_pad_n}},
        {{is_two_pass}}>;

    using one_pass_pipeline_{{idx}} = ck_tile::SmoothquantPipelineOnePass<pipeline_problem_{{idx}}>;
    using two_pass_pipeline_{{idx}} = ck_tile::SmoothquantPipelineTwoPass<pipeline_problem_{{idx}}>;
    using pipeline_{{idx}}        = std::conditional_t<{{is_two_pass}}, two_pass_pipeline_{{idx}}, one_pass_pipeline_{{idx}}>;
    using {{name}} = ck_tile::MoeSmoothquant<pipeline_{{idx}}>;
)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"name", GetInstanceName()},
                                {"idx", idx++},
                                {"shape", tile_desc_.Emit()},
                                {"is_pad_n", is_pad_n_},
                                {"is_two_pass", is_two_pass_}
                               };


    return TEMPLATE_CHECK(tpl, value_map, "MoeSmoothQuantCodeGen::Emit");
}

}  // namespace flashck