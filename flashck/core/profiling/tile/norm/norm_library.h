#pragma once

#include <unordered_map>

namespace flashck {

enum class NormKind {
    LayerNorm = 0,
    RMSNorm   = 1,
};

struct NormTag {
    std::string name;
    std::string problem_tag;
    std::string trait_tag;
    std::string fwd_tag;
    std::string pass_tag;
};

static const std::unordered_map<NormKind, NormTag> norm_map = {
    {NormKind::LayerNorm,
     {"layer_norm",
      "Layernorm2dFwdPipelineProblem",
      "Layernorm2dFwdTraits",
      "Layernorm2dFwd",
      "Layernorm2dFwdPipelineTwoPass"}},
    {NormKind::RMSNorm,
     {"rms_norm",
      "Rmsnorm2dFwdPipelineProblem",
      "Rmsnorm2dFwdTraits",
      "Rmsnorm2dFwd",
      "Layernorm2dFwdPipelineOnePass"}},
};

enum class NormBiasEnum {
    NO_BIAS = 0,
    // add bias before fused add
    ADD_BIAS = 1,
};

struct NormBiasInfo {
    std::string name;
    std::string short_name;
};

static const std::unordered_map<NormBiasEnum, NormBiasInfo> norm_bias_map = {
    {NormBiasEnum::NO_BIAS, {"no_bias", "nb"}},
    {NormBiasEnum::ADD_BIAS, {"add_bias", "ab"}},
};

enum class FusedAddEnum {
    NO_ADD        = 0,
    PRE_ADD_STORE = 1,  // fused add before layernorm and store result to global
    PRE_ADD       = 2,  //  fused add before layernorm, but not store result
};

struct FusedAddInfo {
    std::string name;
    std::string short_name;
};

static const std::unordered_map<FusedAddEnum, FusedAddInfo> fused_add_map = {
    {FusedAddEnum::NO_ADD, {"no_add", "na"}},
    {FusedAddEnum::PRE_ADD_STORE, {"pre_add_store", "pas"}},
    {FusedAddEnum::PRE_ADD, {"pre_add", "pa"}},
};

enum class FusedQuantEnum {
    NO_SWEEP             = 0,
    SMOOTH_DYNAMIC_QUANT = 1,  // smooth oulier + rowwise quant, need input x-scale and store y_scale
    DYNAMIC_QUANT        = 2,  // rowwise quant, store out a y-scale
};

struct FusedQuantInfo {
    std::string name;
    std::string short_name;
};

static const std::unordered_map<FusedQuantEnum, FusedQuantInfo> fused_quant_map = {
    {FusedQuantEnum::NO_SWEEP, {"no_sweep", "ns"}},
    {FusedQuantEnum::SMOOTH_DYNAMIC_QUANT, {"smooth_dynamic_quant", "sdq"}},
    {FusedQuantEnum::DYNAMIC_QUANT, {"dynamic_quant", "dq"}},
};

}  // namespace flashck