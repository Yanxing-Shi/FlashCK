#pragma once

#include "core/profiling/tile/fmha/fmha_fwd_codegen.h"

#include "core/utils/common.h"

namespace flashck {

const std::vector<flashck::FmhaConfig> g_backup_legacy_fmha_config = {
    FmhaConfig{FmhaTileConfig{
        flashck::BlockConfig{
            flashck::IntEnumConfigParam{{{256}}},
            flashck::IntEnumConfigParam{{{128}}},
            flashck::IntEnumConfigParam{{{128}}}
        },
        flashck::WarpConfig{
            flashck::IntEnumConfigParam{{{4}}},
            flashck::IntEnumConfigParam{{{1}}},
            flashck::IntEnumConfigParam{{{1}}}
        },
        flashck::WarpTileConfig{
            flashck::IntEnumConfigParam{{{64}}},
            flashck::IntEnumConfigParam{{{32}}},
            flashck::IntEnumConfigParam{{{32}}}
        }
    },
    FmhaPaddingConfig{
        flashck::BoolEnumConfigParam{{false}},
        flashck::BoolEnumConfigParam{{false}},
        flashck::BoolEnumConfigParam{{false}}
    },
    FmhaLaunchConfig{
        flashck::IntEnumConfigParam{{{1}}},
    },
    FmhaPipelineConfig{
        flashck::StrEnumConfigParam{{"qr_ks_vs"}},
    }}
};

// Generate all possible FmhaFwdCodeGen instances from a FmhaConfig
inline std::vector<FmhaFwdCodeGen> GenerateFmhaInstances(const FmhaFwdCodeGen& config, const GemmProblem& gemm_problem) {
    std::vector<FmhaFwdCodeGen> result;

    // Helper to flatten nested vectors (e.g., vector<vector<int64_t>>) into a single vector
    auto flatten = [](const auto& v) {
        using Elem = typename std::decay_t<decltype(v)>::value_type::value_type;
        std::vector<Elem> out;
        for (const auto& inner : v) out.insert(out.end(), inner.begin(), inner.end());
        return out;
    };

    std::vector<std::vector<ProductElem>> product_lists = {
        // BlockConfig
        flatten(config.tile_config_.block_.m_.values_),
        flatten(config.tile_config_.block_.n_.values_),
        flatten(config.tile_config_.block_.k_.values_),
        // WarpConfig
        flatten(config.tile_config_.warp_.m_.values_),
        flatten(config.tile_config_.warp_.n_.values_),
        flatten(config.tile_config_.warp_.k_.values_),
        // WarpTileConfig
        flatten(config.tile_config_.warp_tile_.m_.values_),
        flatten(config.tile_config_.warp_tile_.n_.values_),
        flatten(config.tile_config_.warp_tile_.k_.values_),
        // PaddingConfig (convert bool to int64_t)
        std::vector<ProductElem>(flatten(config.padding_.m_.values_).begin(), flatten(config.padding_.m_.values_).end()),
        std::vector<ProductElem>(flatten(config.padding_.n_.values_).begin(), flatten(config.padding_.n_.values_).end()),
        std::vector<ProductElem>(flatten(config.padding_.k_.values_).begin(), flatten(config.padding_.k_.values_).end()),
        // LaunchConfig
        flatten(config.launch_.max_block_per_cu_.values_),
        // PipelineConfig (string)
        std::vector<ProductElem>(config.pipeline_.pipeline_.values_.begin(), config.pipeline_.pipeline_.values_.end())
    };

    CartesianProduct(product_lists, [&](const std::vector<flashck::ProductElem>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t block_m = std::get<int64_t>(vals[idx++]);
        int64_t block_n = std::get<int64_t>(vals[idx++]);
        int64_t block_k = std::get<int64_t>(vals[idx++]);
        // WarpConfig
        int64_t warp_m = std::get<int64_t>(vals[idx++]);
        int64_t warp_n = std::get<int64_t>(vals[idx++]);
        int64_t warp_k = std::get<int64_t>(vals[idx++]);
        // WarpTileConfig
        int64_t warptile_m = std::get<int64_t>(vals[idx++]);
        int64_t warptile_n = std::get<int64_t>(vals[idx++]);
        int64_t warptile_k = std::get<int64_t>(vals[idx++]);
        // PaddingConfig
        bool pad_m = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool pad_n = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool pad_k = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        // LaunchConfig
        int64_t max_block_per_cu = std::get<int64_t>(vals[idx++]);
        // PipelineConfig
        std::string pipeline = std::get<std::string>(vals[idx++]);

        // Construct FmhaFwdCodeGen
        FmhaFwdCodeGen fmha;
        fmha.problem_ = gemm_problem;
        // Fill tile_desc_
        fmha.tile_desc_.bm0_ = block_m;
        fmha.tile_desc_.bn0_ = block_n;
        fmha.tile_desc_.bk0_ = block_k;
        fmha.tile_desc_.rm0_ = warp_m;
        fmha.tile_desc_.rn0_ = warp_n;
        fmha.tile_desc_.rk0_ = warp_k;
        fmha.tile_desc_.wm0_ = warptile_m;
        fmha.tile_desc_.wn0_ = warptile_n;
        fmha.tile_desc_.wk0_ = warptile_k;
        // Padding
        fmha.is_pad_q_seq_len_ = pad_m;
        fmha.is_pad_kv_seq_len_ = pad_n;
        fmha.is_pad_qk_head_dim_ = pad_k;
        // Launch
        fmha.block_per_cu_ = static_cast<int>(max_block_per_cu);
        // Pipeline
        fmha.pipeline_ = 
        result.push_back(fmha);

    
    
    });

    return result;
}




} // namespace flashck