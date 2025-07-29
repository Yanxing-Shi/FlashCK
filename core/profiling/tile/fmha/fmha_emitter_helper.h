#pragma once

#include "core/profiling/tile/fmha/fmha_fwd_codegen.h"

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaFwdConfig> g_backup_fmha_config = {
    FmhaFwdConfig{FmhaFwdTileConfig{
        FmhaFwdBlockConfig{
            IntEnumConfigParam{{{256}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}}
        },
        FmhaFwdWarpConfig{
            IntEnumConfigParam{{{4}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{1}}}
        },
        FmhaFwdWarpTileConfig{
            IntEnumConfigParam{{{64}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{32}}}
        }
    },
    FmhaFwdPaddingConfig{
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}}
    },
    FmhaFwdLaunchConfig{
        IntEnumConfigParam{{{1}}},
    },
    StrEnumConfigParam{{"qr_ks_vs"}},
    }
};

// Generate all possible FmhaFwdCodeGen instances from a FmhaFwdConfig
inline std::vector<FmhaFwdCodeGen> GenerateFmhaInstances(const FmhaFwdConfig& config, const FmhaProblem& fmha_problem) {
    std::vector<FmhaFwdCodeGen> result;

    // Helper to flatten vector<vector<T>> or just return vector<T> as is
    auto flatten = [](const auto& v) {
        // Specialize for StrEnumConfigParam and similar types
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, StrEnumConfigParam>) {
            return v.values_;
        } else {
            using VecT = std::decay_t<decltype(v)>;
            using ElemT = typename VecT::value_type;
            if constexpr (std::is_same_v<ElemT, bool> || std::is_arithmetic_v<ElemT> || std::is_same_v<ElemT, std::string>) {
                // v is vector<T>, just return as is
                return std::vector<ElemT>(v.begin(), v.end());
            } else {
                // v is vector<vector<T>>
                std::vector<typename ElemT::value_type> out;
                for (const auto& inner : v) out.insert(out.end(), inner.begin(), inner.end());
                return out;
            }
        }
    };

    using ProductElem = std::variant<int64_t, std::vector<int64_t>>;
    std::vector<std::vector<ProductElem>> all_lists = {
        // BlockConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.m0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.n0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.k0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.k0_max_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.n1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.k1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.m0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.n0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.k0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
         [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.m1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.n1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.k1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpTileConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.m0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.n0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.k0_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.m1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.n1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.k1_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding_.s_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding_.sk_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding_.d_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding_.dv_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.launch_.min_block_per_cu_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PipelineConfig (enum as int64_t)
        [&]{ std::vector<ProductElem> v; for (const auto& x : flatten(config.pipeline_.values_)) v.emplace_back(static_cast<int64_t>(GetBlockFmhaPipelineEnumFromString(x))); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<flashck::ProductElem>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t m0_block = std::get<int64_t>(vals[idx++]);
        int64_t n0_block = std::get<int64_t>(vals[idx++]);
        int64_t k0_block = std::get<int64_t>(vals[idx++]);
        int64_t n1_block = std::get<int64_t>(vals[idx++]);
        int64_t k1_block = std::get<int64_t>(vals[idx++]);
        int64_t k0_max_block = std::get<int64_t>(vals[idx++]);

        int64_t m0_warp = std::get<int64_t>(vals[idx++]);
        int64_t n0_warp = std::get<int64_t>(vals[idx++]);
        int64_t k0_warp = std::get<int64_t>(vals[idx++]);
        int64_t m1_warp = std::get<int64_t>(vals[idx++]);
        int64_t n1_warp = std::get<int64_t>(vals[idx++]);
        int64_t k1_warp = std::get<int64_t>(vals[idx++]);

        int64_t m0_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t n0_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t k0_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t m1_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t n1_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t k1_warp_tile = std::get<int64_t>(vals[idx++]);

        // PaddingConfig
        bool is_pad_q_seq_len_ = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_kv_seq_len_ = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_qk_head_dim_ = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_v_head_dim_ = static_cast<bool>(std::get<int64_t>(vals[idx++]));

        // launch config
        int64_t min_block_per_cu = std::get<int64_t>(vals[idx++]);

        // PipelineConfig
        BlockFmhaPipelineEnum pipeline = static_cast<BlockFmhaPipelineEnum>(std::get<int64_t>(vals[idx++]));

        // Construct FmhaFwdCodeGen
        FmhaFwdCodeGen fmha;
        fmha.problem_ = fmha_problem;
        // tile_desc
        fmha.tile_desc_.m0_block_ = m0_block;
        fmha.tile_desc_.n0_block_ = n0_block;
        fmha.tile_desc_.k0_block_ = k0_block;
        fmha.tile_desc_.k0_max_block_ = k0_max_block;
        fmha.tile_desc_.n1_block_ = n1_block;
        fmha.tile_desc_.k1_block_ = k1_block;
        fmha.tile_desc_.m0_warp_ = m0_warp;
        fmha.tile_desc_.n0_warp_ = n0_warp;
        fmha.tile_desc_.k0_warp_ = k0_warp;
        fmha.tile_desc_.m1_warp_ = m1_warp;
        fmha.tile_desc_.n1_warp_ = n1_warp;
        fmha.tile_desc_.k1_warp_ = k1_warp;
        fmha.tile_desc_.m0_warp_tile_ = m0_warp_tile;
        fmha.tile_desc_.n0_warp_tile_ = n0_warp_tile;
        fmha.tile_desc_.k0_warp_tile_ = k0_warp_tile;
        fmha.tile_desc_.m1_warp_tile_ = m1_warp_tile;
        fmha.tile_desc_.n1_warp_tile_ = n1_warp_tile;
        fmha.tile_desc_.k1_warp_tile_ = k1_warp_tile;
        // Padding
        fmha.is_pad_q_seq_len_ = is_pad_q_seq_len_;
        fmha.is_pad_kv_seq_len_ = is_pad_kv_seq_len_;
        fmha.is_pad_qk_head_dim_ = is_pad_qk_head_dim_;
        fmha.is_pad_v_head_dim_ = is_pad_v_head_dim_;
        // Launch
        fmha.min_block_per_cu_ = min_block_per_cu;
        // Pipeline
        fmha.pipeline_ = pipeline;
        result.push_back(fmha);
    });

    return result;
}




} // namespace flashck