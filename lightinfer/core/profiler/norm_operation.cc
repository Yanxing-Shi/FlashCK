#include "lightinfer/core/profiler/norm_operation.h"

#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/jinjia2_utils.h"
#include "lightinfer/core/utils/printf.h"

#include <hip/hip_runtime.h>

namespace lightinfer {

std::string NormTileDesc::GetConfigName()
{
    return Sprintf("{}_{}_{}_{}_{}", repeat_m_, repeat_n_, thread_per_block_m_, thread_per_block_n_, vector_n_);
}

std::string NormTileDesc::Emit()
{
    bool is_warp_per_row = thread_per_block_n_ <= warpSize;
    LI_ENFORCE_EQ((thread_per_block_m_ * thread_per_block_n_) % warpSize,
                  0,
                  Unavailable("thread_per_block_m_ * thread_per_block_n_ must be multiple of warpSize"));

    int64_t total_warps = (thread_per_block_m_ * thread_per_block_n_) / warpSize;
    // num of warps along m
    int64_t block_warps_m = [&]() -> int64_t {
        if (is_warp_per_row) {
            LI_ENFORCE_EQ(
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
            LI_ENFORCE_EQ(
                warpSize % thread_per_block_n_, 0, Unavailable("thread_per_block_n_ must be multiple of warpSize"));
            return 1;
        }
        else {
            LI_ENFORCE_EQ(
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

    return TemplateLoadAndRender(tile_desc, tile_desc_value_map);
}

std::string NormOperation::GetConfigName()
{
    if (fused_add_ == FusedAddEnum::PRE_ADD_STORE) {
        epilogue_op_ = TensorOperation::PreAddStore;
    }
    else if (fused_add_ == FusedAddEnum::PRE_ADD) {
        epilogue_op_ = TensorOperation::PreAdd;
    }
    else {
        epilogue_op_ = TensorOperation::PassThrough;
    }

    return Sprintf("{}_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                   g_norm_operation_kind_names_map.find(operation_kind_)->second,
                   g_short_tensor_operation_names_map.find(epilogue_op_)->second,
                   DataTypeToShortString(x_dtype_),
                   DataTypeToShortString(y_dtype_),
                   DataTypeToShortString(smooth_scale_dtype_),
                   DataTypeToShortString(y_scale_dtype_),
                   tile_desc_.GetConfigName(),
                   g_tile_layer_norm_operation_kind_short_names_map.find(is_add_bias_)->second,
                   g_fused_add_enum_str_map.find(fused_add_)->second,
                   g_fused_quant_enum_str_map.find(fused_quant_)->second);
}

std::string NormOperation::Emit()
{
    std::string source = R"(
using PipelineProblem_{{idx}} = ck_tile::{{norm_problem}}<
    XDataType,
{% if norm_kind == "layer_norm" %}
    XBiasDataType,
{% endif %}
    GammaDataType,
    BetaDataType,
    ComputeDataType,
    YDataType,
    MeanDataType,
    InvStdDataType,
    SmoothScaleDataType,
    YScaleDataType,
    {{shape}}
    ck_tile::{{norm_traits}}<{{is_pad_n}} /*kPadN*/,
                                    false /*kSaveMeanInvStd*/,
                                {% if norm_kind == "layer_norm" %}
                                    {{is_fast_div}} /*kFastFDiv*/,
                                    {{is_welford}} /*kWelford*/,
                                {% endif %}
                                    {{is_two_pass}} /*kTwoPass*/,
                                {% if norm_kind == "layer_norm" %}
                                    static_cast<ck_tile::Layernorm2dXBiasEnum>({{is_add_bias}}) /*kisaddbias*/,
                                {% endif %}
                                    static_cast<ck_tile::Layernorm2dFusedAddEnum>({{fused_add}}) /*kFusedAdd*/,
                                    static_cast<ck_tile::Layernorm2dFusedQuantEnum>({{fused_quant}}) /*kFusedQuant*/>>;

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
    static int  idx    = 0;

    jinja2::ValuesMap value_map{{"name", GetConfigName()},
                                {"idx", idx++},
                                {"norm_kind", g_norm_operation_kind_names_map.at(operation_kind_)},
                                {"norm_problem", g_norm_operation_problem_tag_map.at(operation_kind_)},
                                {"norm_traits", g_norm_operation_trait_tag_map.at(operation_kind_)},
                                {"norm_fwd", g_norm_operation_fwd_tag_map.at(operation_kind_)},
                                {"norm_pass", g_norm_operation_pass_tag_map.at(operation_kind_)},
                                {"is_pad_n", "true"},
                                {"is_fast_div", "true"},
                                {"is_two_pass", "true"},
                                {"is_welford", "true"},
                                {"is_add_bias", static_cast<int>(is_add_bias_)},
                                {"fused_add", static_cast<int>(fused_add_)},
                                {"fused_quant", static_cast<int>(fused_quant_)},
                                {"is_smooth_quant", static_cast<int>(fused_quant_) == 1 ? true : false},
                                {"shape", tile_desc_.Emit()}};

    return TemplateLoadAndRender(source, value_map);
}

}  // namespace lightinfer