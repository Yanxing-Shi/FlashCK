#include "core/profiling/tile/norm/norm_codegen.h"

namespace flashck {

std::string NormTileDesc::GetInstanceName() const
{
    return Sprintf("{repeat_m}_{repeat_n}_{thread_per_block_m}_{thread_per_block_n}_{vector_n}",
                   fmt::arg("repeat_m", repeat_m_),
                   fmt::arg("repeat_n", repeat_n_),
                   fmt::arg("thread_per_block_m", thread_per_block_m_),
                   fmt::arg("thread_per_block_n", thread_per_block_n_),
                   fmt::arg("vector_n", vector_n_));
}

std::string NormTileDesc::Emit() const
{
    bool is_warp_per_row = thread_per_block_n_ <= warpSize;
    FC_ENFORCE_EQ((thread_per_block_m_ * thread_per_block_n_) % warpSize,
                  0,
                  Unavailable("thread_per_block_m_ * thread_per_block_n_ must be multiple of warpSize"));

    int64_t total_warps = (thread_per_block_m_ * thread_per_block_n_) / warpSize;
    // num of warps along m
    int64_t block_warps_m = [&]() -> int64_t {
        if (is_warp_per_row) {
            FC_ENFORCE_EQ(
                warpSize % thread_per_block_n_, 0, Unavailable("thread_per_block_n_ must be multiple of warpSize"));
            return total_warps * (warpSize / thread_per_block_n_);
        }
        else {
            // static_assert(warpSize % thread_per_block_m_ == 0);
            return total_warps / (thread_per_block_n_ / warpSize);
        }
    }();

    // num of warps along n
    int64_t block_warps_n = [&]() -> int64_t {
        if (is_warp_per_row) {
            FC_ENFORCE_EQ(
                warpSize % thread_per_block_n_, 0, Unavailable("thread_per_block_n_ must be multiple of warpSize"));
            return 1;
        }
        else {
            FC_ENFORCE_EQ(
                thread_per_block_n_ % warpSize, 0, Unavailable("thread_per_block_n_ must be multiple of warpSize"));

            return thread_per_block_n_ / warpSize;
        }
    }();

    int64_t block_m = repeat_m_ * thread_per_block_m_;
    int64_t block_n = repeat_n_ * thread_per_block_n_ * vector_n_;

    int64_t warp_m = thread_per_block_m_ / block_warps_m;
    int64_t warp_n = thread_per_block_n_ / block_warps_n * vector_n_;

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
        {"vector_n", vector_n_},
    };

    return TEMPLATE_CHECK(tile_desc, tile_desc_value_map, "NormTileDesc::Emit");
}

std::string NormCodeGen::GetInstanceName() const
{
    return Sprintf("{kind_name}_{x_dtype}_{y_dtype}_{smooth_scale_dtype}_{y_scale_dtype}_"
                   "{tile_desc}_{is_add_bias}_{fused_add}_{fused_quant}",
                   fmt::arg("kind_name", GetNormKindShortName(kind_)),
                   fmt::arg("x_dtype", DataTypeToString(x_dtype_)),
                   fmt::arg("y_dtype", DataTypeToString(y_dtype_)),
                   fmt::arg("smooth_scale_dtype", DataTypeToString(smooth_scale_dtype_)),
                   fmt::arg("y_scale_dtype", DataTypeToString(y_scale_dtype_)),
                   fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
                   fmt::arg("is_add_bias", GetNormBiasShortName(is_add_bias_)),
                   fmt::arg("fused_add", GetFusedAddShortName(fused_add_)),
                   fmt::arg("fused_quant", GetFusedQuantShortName(fused_quant_)));
}

std::string NormCodeGen::Emit() const
{
    std::string tpl = R"(
using PipelineProblem_{{idx}} = ck_tile::{{norm_problem}}<
    XDataType,
{% if norm_kind == "layer_norm" %}
    XBiasDataType,
{% endif %}
    GammaDataType,
{% if norm_kind == "layer_norm" %}
    BetaDataType,
{% endif %}
    ComputeDataType,
    YDataType,
{% if norm_kind == "layer_norm" %}
    MeanDataType,
{% endif %}
{% if norm_kind == "layer_norm" %}
    InvStdDataType,
{% else %}
    InvRmsDataType,
{% endif %}
{% if norm_kind == "rms_norm" %}
    UnquantYDataType,
{% endif %}
    SmoothScaleDataType,
    YScaleDataType,
    {{shape}}
    ck_tile::{{norm_traits}}<{{is_pad_n}} /*kPadN*/,
                                    false /*kSaveMeanInvStd*/,
                                {% if norm_kind == "layer_norm" %}
                                    {{is_fast_div}} /*kFastFDiv*/,
                                    {{is_welford}} /*kWelford*/,
                                {% else %}
                                    false /*kSaveUnquantY*/,
                                {% endif %}
                                    {{is_two_pass}} /*kTwoPass*/,
                                {% if norm_kind == "layer_norm" %}
                                    static_cast<ck_tile::Layernorm2dXBiasEnum>({{is_add_bias}}) /*kisaddbias*/,
                                    static_cast<ck_tile::Layernorm2dFusedAddEnum>({{fused_add}}) /*kFusedAdd*/,
                                    static_cast<ck_tile::Layernorm2dFusedQuantEnum>({{fused_quant}}) /*kFusedQuant*/>>;
                                {% else %}
                                    static_cast<ck_tile::Rmsnorm2dFusedAddEnum>({{fused_add}}) /*kFusedAdd*/,
                                    static_cast<ck_tile::Rmsnorm2dFusedQuantEnum>({{fused_quant}}) /*kFusedQuant*/,
                                    static_cast<ck_tile::Rmsnorm2dSensitiveEnum>(0) /*USEModelSensitive*/>>;
                                {% endif %}

{% if is_smooth_quant %}
using DynamicQuantEpilogueProblem_{{idx}}         = ck_tile::DynamicQuantEpilogueProblem<
    ComputeDataType,
    SmoothScaleDataType,
    YScaleDataType,
    YDataType,
    {{shape}}
    ck_tile::DynamicQuantEpilogueTraits<false, true, {{is_smooth_quant}}, false, true /*max3*/>>;
{% else %}
using Default2DEpilogueProblem_{{idx}} =
    ck_tile::Default2DEpilogueProblem<ComputeDataType, YDataType, false, {{is_pad_n}} /*kPadN*/, false>;
{% endif %}

using {{name}} = ck_tile::{{norm_fwd}}<
{% if is_two_pass %}
    ck_tile::{{norm_pass}}<PipelineProblem_{{idx}}>,
{% else %}
    ck_tile::{{norm_pass}}<PipelineProblem_{{idx}}>,
{% endif %}

{% if is_smooth_quant %}
    ck_tile::DynamicQuantEpilogue<DynamicQuantEpilogueProblem_{{idx}}>
{% else %}
    ck_tile::Default2DEpilogue<Default2DEpilogueProblem_{{idx}}>
{% endif %}
    >;

)";
    static int  idx = 0;

    jinja2::ValuesMap value_map{{"name", GetInstanceName()},
                                {"idx", idx++},
                                {"norm_kind", GetNormKindName(kind_)},
                                {"norm_problem", GetNormKindProblemTag(kind_)},
                                {"norm_traits", GetNormKindTraitTag(kind_)},
                                {"norm_fwd", GetNormKindFwdTag(kind_)},
                                {"norm_pass", GetNormKindPassTag(kind_)},
                                {"is_pad_n", "true"},
                                {"is_fast_div", "true"},
                                {"is_two_pass", "true"},
                                {"is_welford", "true"},
                                {"is_add_bias", static_cast<int>(is_add_bias_)},
                                {"fused_add", static_cast<int>(fused_add_)},
                                {"fused_quant", static_cast<int>(fused_quant_)},
                                {"is_smooth_quant", static_cast<int>(fused_quant_) == 1 ? true : false},
                                {"shape", tile_desc_.Emit()}};

    return TEMPLATE_CHECK(tpl, value_map, "NormCodeGen::Emit");
}

}  // namespace flashck