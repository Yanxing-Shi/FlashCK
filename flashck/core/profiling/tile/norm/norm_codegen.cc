#include "flashck/core/profiling/tile/norm/norm_codegen.h"

#include <algorithm>
#include <sstream>

namespace flashck {

// Static member initialization
int NormCodeGen::instance_counter_ = 0;

// ==================== NormTileDesc Implementation ====================

std::string NormTileDesc::GetConfigName() const
{
    return fmt::format("{}_{}_{}_{}_{}", repeat_m_, repeat_n_, thread_per_block_m_, thread_per_block_n_, vector_n_);
}

std::string NormTileDesc::ToString() const
{
    return fmt::format("NormTileDesc(repeat_m={}, repeat_n={}, "
                       "thread_per_block_m={}, thread_per_block_n={}, vector_n={})",
                       repeat_m_,
                       repeat_n_,
                       thread_per_block_m_,
                       thread_per_block_n_,
                       vector_n_);
}

bool NormTileDesc::IsValid() const
{
    // Check for positive values
    if (repeat_m_ <= 0 || repeat_n_ <= 0 || thread_per_block_m_ <= 0 || thread_per_block_n_ <= 0 || vector_n_ <= 0) {
        return false;
    }

    // Check thread block size limits
    const int64_t total_threads = GetTotalThreads();
    if (total_threads > 1024) {  // Common GPU limit
        return false;
    }

    // Check vector size constraints
    if (vector_n_ > thread_per_block_n_) {
        return false;
    }

    // Check warp size alignment
    constexpr int64_t warpSize = 32;
    if (total_threads % warpSize != 0) {
        return false;
    }

    return true;
}

std::pair<int64_t, int64_t> NormTileDesc::CalculateWarpDistribution() const
{
    constexpr int64_t warpSize        = 32;
    const bool        is_warp_per_row = thread_per_block_n_ <= warpSize;
    const int64_t     total_warps     = GetTotalThreads() / warpSize;

    int64_t block_warps_m, block_warps_n;

    if (is_warp_per_row) {
        FC_ENFORCE_EQ(warpSize % thread_per_block_n_, 0, "thread_per_block_n_ must be divisor of warpSize");
        block_warps_m = total_warps * (warpSize / thread_per_block_n_);
        block_warps_n = 1;
    }
    else {
        FC_ENFORCE_EQ(thread_per_block_n_ % warpSize, 0, "thread_per_block_n_ must be multiple of warpSize");
        block_warps_m = total_warps / (thread_per_block_n_ / warpSize);
        block_warps_n = thread_per_block_n_ / warpSize;
    }

    return {block_warps_m, block_warps_n};
}

std::string NormTileDesc::Emit() const
{
    if (!IsValid()) {
        FC_THROW(std::invalid_argument("Invalid tile descriptor: " + ToString()));
    }

    constexpr int64_t warpSize = 32;
    FC_ENFORCE_EQ(GetTotalThreads() % warpSize, 0, "Total threads must be multiple of warpSize");

    auto [block_warps_m, block_warps_n] = CalculateWarpDistribution();

    const int64_t block_m = GetEffectiveM();
    const int64_t block_n = GetEffectiveN();
    const int64_t warp_m  = thread_per_block_m_ / block_warps_m;
    const int64_t warp_n  = thread_per_block_n_ / block_warps_n * vector_n_;

    const std::string tile_template = R"(
    ck_tile::Generic2dBlockShape<ck_tile::sequence<{}, {}>,
                                ck_tile::sequence<{}, {}>,
                                ck_tile::sequence<{}, {}>, 
                                ck_tile::sequence<1, {}>>,
)";

    return fmt::format(tile_template, block_m, block_n, block_warps_m, block_warps_n, warp_m, warp_n, vector_n_);
}

// ==================== NormCodeGen Implementation ====================

std::string NormCodeGen::GetConfigName() const
{
    return fmt::format("{}_{}_{}_{}_{}_{}_{}_{}_{}",
                       GetNormKindName(kind_),
                       DataTypeToString(x_dtype_),
                       DataTypeToString(y_dtype_),
                       DataTypeToString(smooth_scale_dtype_),
                       DataTypeToString(y_scale_dtype_),
                       tile_desc_.GetConfigName(),
                       GetBiasName(is_add_bias_),
                       GetFusedAddName(fused_add_),
                       GetFusedQuantName(fused_quant_));
}

std::string NormCodeGen::ToString() const
{
    return fmt::format("NormCodeGen(kind={}, x_dtype={}, y_dtype={}, "
                       "smooth_scale_dtype={}, y_scale_dtype={}, "
                       "tile_desc={}, is_add_bias={}, fused_add={}, fused_quant={})",
                       GetNormKindName(kind_),
                       DataTypeToString(x_dtype_),
                       DataTypeToString(y_dtype_),
                       DataTypeToString(smooth_scale_dtype_),
                       DataTypeToString(y_scale_dtype_),
                       tile_desc_.ToString(),
                       GetBiasName(is_add_bias_),
                       GetFusedAddName(fused_add_),
                       GetFusedQuantName(fused_quant_));
}

bool NormCodeGen::IsValid() const
{
    // Validate enum values
    if (!IsValidNormKind(kind_) || !IsValidBiasEnum(is_add_bias_) || !IsValidFusedAddEnum(fused_add_)
        || !IsValidFusedQuantEnum(fused_quant_)) {
        return false;
    }

    // Validate tile descriptor
    if (!tile_desc_.IsValid()) {
        return false;
    }

    // Validate data types
    if (!ValidateDataTypes()) {
        return false;
    }

    // Validate fusion modes
    if (!ValidateFusionModes()) {
        return false;
    }

    return true;
}

bool NormCodeGen::ValidateDataTypes() const
{
    // Check if data types are supported
    const std::vector<DataType> supported_types = {DataType::Float16, DataType::Float32, DataType::BFloat16};

    auto is_supported = [&](DataType type) {
        return std::find(supported_types.begin(), supported_types.end(), type) != supported_types.end();
    };

    return is_supported(x_dtype_) && is_supported(y_dtype_) && is_supported(smooth_scale_dtype_)
           && is_supported(y_scale_dtype_);
}

bool NormCodeGen::ValidateFusionModes() const
{
    // RMSNorm doesn't support bias
    if (kind_ == NormKind::RMSNorm && is_add_bias_ != NormBiasEnum::NO_BIAS) {
        return false;
    }

    // Smooth quantization requires specific scale types
    if (fused_quant_ == FusedQuantEnum::SMOOTH_DYNAMIC_QUANT) {
        if (smooth_scale_dtype_ != DataType::Float32) {
            return false;
        }
    }

    return true;
}

std::string NormCodeGen::GetTemplateSource() const
{
    return R"(
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
                                    static_cast<ck_tile::Layernorm2dXBiasEnum>({{is_add_bias}}) /*kIsAddBias*/,
                                {% endif %}
                                    static_cast<ck_tile::Layernorm2dFusedAddEnum>({{fused_add}}) /*kFusedAdd*/,
                                    static_cast<ck_tile::Layernorm2dFusedQuantEnum>({{fused_quant}}) /*kFusedQuant*/>>;

{% if is_smooth_quant %}
using DynamicQuantEpilogueProblem_{{idx}} = ck_tile::DynamicQuantEpilogueProblem<
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
}

jinja2::ValuesMap NormCodeGen::GenerateValueMap() const
{
    const auto& norm_info       = norm_map.at(kind_);
    const bool  is_smooth_quant = (fused_quant_ == FusedQuantEnum::SMOOTH_DYNAMIC_QUANT);
    const bool  is_two_pass     = (kind_ == NormKind::LayerNorm);

    return jinja2::ValuesMap{{"name", GetConfigName()},
                             {"idx", instance_counter_++},
                             {"norm_kind", norm_info.name},
                             {"norm_problem", norm_info.problem_tag},
                             {"norm_traits", norm_info.trait_tag},
                             {"norm_fwd", norm_info.fwd_tag},
                             {"norm_pass", norm_info.pass_tag},
                             {"shape", tile_desc_.Emit()},
                             {"is_pad_n", true},
                             {"is_fast_div", true},
                             {"is_two_pass", is_two_pass},
                             {"is_welford", true},
                             {"is_add_bias", static_cast<int>(is_add_bias_)},
                             {"fused_add", static_cast<int>(fused_add_)},
                             {"fused_quant", static_cast<int>(fused_quant_)},
                             {"is_smooth_quant", is_smooth_quant}};
}

std::string NormCodeGen::Emit() const
{
    if (!IsValid()) {
        FC_THROW(std::invalid_argument("Invalid norm code generation configuration: " + ToString()));
    }

    const std::string       template_source = GetTemplateSource();
    const jinja2::ValuesMap value_map       = GenerateValueMap();

    try {
        return TemplateLoadAndRender(template_source, value_map);
    }
    catch (const std::exception& e) {
        FC_THROW(std::runtime_error("Template rendering failed: " + std::string(e.what())));
    }
}

}  // namespace flashck