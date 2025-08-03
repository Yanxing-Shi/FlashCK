#include "core/profiling/norm/rms_norm/rms_norm_codegen.h"

namespace flashck {

std::string RmsNormTileDesc::GetInstanceName() 
{
    return Sprintf("{m_repeat}_{n_repeat}_{m_thread_per_block}_{n_thread_per_block}_{n_vector}",
                   fmt::arg("m_repeat", m_repeat_),
                   fmt::arg("n_repeat", n_repeat_),
                   fmt::arg("m_thread_per_block", m_thread_per_block_),
                   fmt::arg("n_thread_per_block", n_thread_per_block_),
                   fmt::arg("n_vector", n_vector_));
}

std::string RmsNormTileDesc::Emit() 
{
    bool is_warp_per_row = n_thread_per_block_ <= warpSize;
    FC_ENFORCE_EQ((m_thread_per_block_ * n_thread_per_block_) % warpSize,
                  0,
                  Unavailable("m_thread_per_block_ * n_thread_per_block_ must be multiple of warpSize"));

    int64_t total_warps = (m_thread_per_block_ * n_thread_per_block_) / warpSize;
    // num of warps along m
    int64_t m_block_warps = [&]() -> int64_t {
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
    int64_t n_block_warps = [&]() -> int64_t {
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

    int64_t m_block = m_repeat_ * m_thread_per_block_;
    int64_t n_block = n_repeat_ * n_thread_per_block_ * n_vector_;

    int64_t m_warp = m_thread_per_block_ / m_block_warps;
    int64_t n_warp = n_thread_per_block_ / n_block_warps * n_vector_;

    std::string tile_desc = R"(
    ck_tile::Generic2dBlockShape<ck_tile::sequence<{{m_block}}, {{n_block}}>,
                                ck_tile::sequence<{{m_block_warps}}, {{n_block_warps}}>,
                                ck_tile::sequence<{{m_warp}}, {{n_warp}}>, 
                                ck_tile::sequence<1, {{n_vector}}>>,
)";

    jinja2::ValuesMap tile_desc_value_map = {
        {"m_block", m_block},
        {"n_block", n_block},
        {"m_block_warps", m_block_warps},
        {"n_block_warps", n_block_warps},
        {"m_warp", m_warp},
        {"n_warp", n_warp},
        {"n_vector", n_vector_},
    };

    return TEMPLATE_CHECK(tile_desc, tile_desc_value_map, "RmsNormTileDesc::Emit");
}
std::string RmsNormCodeGen::GetInstanceName() 
{
    return Sprintf("rms_norm_{x_dtype}_{y_dtype}_{smooth_scale_dtype}_{y_scale_dtype}_"
                   "{tile_desc}_{is_add_bias}_{fused_add}_{fused_quant}",
                   fmt::arg("x_dtype", DataTypeToString(problem_.x_dtype_)),
                   fmt::arg("y_dtype", DataTypeToString(problem_.y_dtype_)),
                   fmt::arg("smooth_scale_dtype", DataTypeToString(problem_.smooth_scale_dtype_)),
                   fmt::arg("y_scale_dtype", DataTypeToString(problem_.y_scale_dtype_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("is_add_bias", GetNormBiasShortName(problem_.is_add_bias_)),
                   fmt::arg("fused_add", GetFusedAddShortName(problem_.fused_add_)),
                   fmt::arg("fused_quant", GetFusedQuantShortName(problem_.fused_quant_)));
}

std::string RmsNormCodeGen::Emit() 
{
    std::string tpl = R"(
using PipelineProblem_{{idx}} = ck_tile::Rmsnorm2dFwdPipelineProblem<
    XDataType,
    GammaDataType,
    ComputeDataType,
    YDataType,
    InvRmsDataType,
    UnquantYDataType,
    SmoothScaleDataType,
    YScaleDataType,
    {{shape}}
    ck_tile::{{norm_traits}}<{{is_pad_n}} /*kPadN*/,
                                    false /*kSaveMeanInvStd*/,
                                    false /*kSaveUnquantY*/,
                                    {{is_two_pass}} /*kTwoPass*/,
                                    static_cast<ck_tile::Rmsnorm2dFusedAddEnum>({{fused_add}}) /*kFusedAdd*/,
                                    static_cast<ck_tile::Rmsnorm2dFusedQuantEnum>({{fused_quant}}) /*kFusedQuant*/,
                                    static_cast<ck_tile::Rmsnorm2dSensitiveEnum>(0) /*USEModelSensitive*/>>;

    {% if is_smooth_quant %}
        {% if is_save_unquant %}
    using Default2DAndDynamicQuantEpilogueProblem = ck_tile::Default2DAndDynamicQuantEpilogueProblem<
        ComputeDataType, SmoothScaleDataType, YScaleDataType, YDataType, UnquantYDataType, {{Shape}},
        ck_tile::Default2DAndDynamicQuantEpilogueTraits<false, {{is_pad_n}}, {{is_smooth_quant}}, false,  true/*max3*/>>;
        {% else %}
    using DynamicQuantEpilogueProblem_{{idx}}         = ck_tile::DynamicQuantEpilogueProblem<
        ComputeDataType,
        SmoothScaleDataType,
        YScaleDataType,
        YDataType,
        {{shape}}
        ck_tile::DynamicQuantEpilogueTraits<false, true, {{is_smooth_quant}}, false, true /*max3*/>>;
        {% endif %}
    {% else %}
    using Default2DEpilogueProblem_{{idx}} =
        ck_tile::Default2DEpilogueProblem<ComputeDataType, YDataType, false, {{is_pad_n}} /*kPadN*/, false>;
    {% endif %}

    using {{name}} = ck_tile::Rmsnorm2dFwd<
    {% if is_two_pass %}
        ck_tile::Rmsnorm2dFwdPipelineOnePass<PipelineProblem_{{idx}}>,
    {% else %}
        ck_tile::Rmsnorm2dFwdPipelineTwoPass<PipelineProblem_{{idx}}>,
    {% endif %}

    {% if is_smooth_quant %}
        {% if is_save_unquant %}
        ck_tile::Default2DAndDynamicQuantEpilogue<Default2DAndDynamicQuantEpilogueProblem_{{idx}}>;
        {% else %}
        ck_tile::DynamicQuantEpilogue<DynamicQuantEpilogueProblem_{{idx}}>
        {% endif %}
    {% else %}
        ck_tile::Default2DEpilogue<Default2DEpilogueProblem_{{idx}}>
    {% endif %}
    >;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"name", GetInstanceName()},
                                {"idx", idx++},
                                {"is_pad_n", is_pad_n_},
                                {"is_save_unquant", false},
                                {"is_fast_div", true},
                                {"is_two_pass", is_two_pass_},
                                {"is_welford", true},
                                {"is_add_bias", static_cast<int>(problem_.is_add_bias_)},
                                {"fused_add", static_cast<int>(problem_.fused_add_)},
                                {"fused_quant", static_cast<int>(problem_.fused_quant_)},
                                {"is_smooth_quant", static_cast<int>(problem_.fused_quant_) == 1 ? true : false},
                                {"shape", tile_desc_.Emit()}};

    return TEMPLATE_CHECK(tpl, value_map, "RmsNormCodeGen::Emit");
}

}  // namespace flashck