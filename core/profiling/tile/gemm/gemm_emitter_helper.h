#pragma once

#include <variant>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

#include "core/profiling/tile/gemm/gemm_codegen.h"
#include "core/utils/common.h"

namespace flashck {

namespace tile{

const std::vector<flashck::TileGemmConfig> g_tile_gemm_backup_tile_config = {flashck::TileGemmConfig{
    flashck::TileConfig{
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
    flashck::PaddingConfig{
        flashck::BoolEnumConfigParam{{false}},
        flashck::BoolEnumConfigParam{{false}},
        flashck::BoolEnumConfigParam{{false}}
    },
    flashck::LaunchConfig{
        flashck::IntEnumConfigParam{{{1}}}
    },
    flashck::PartitionConfig{
        flashck::IntEnumConfigParam{{{1}}},
        flashck::IntEnumConfigParam{{{1}}},
        flashck::IntEnumConfigParam{{{1}}},
    },
    flashck::PipelineConfig{
        flashck::StrEnumConfigParam{{"compv3"}},
        flashck::StrEnumConfigParam{{"intrawave"}}
    },
    flashck::StrEnumConfigParam{{"cshuffle"}}
}
};


// Generate all possible GemmCodeGen instances from a TileGemmConfig
inline std::vector<GemmCodeGen> GenerateTileGemmInstances(const flashck::TileGemmConfig& config, const GemmProblem& gemm_problem) {
    std::vector<GemmCodeGen> result;

    // Helper to flatten vector<vector<T>> or just return vector<T> as is
    auto flatten = [](const auto& v) {
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
    };

    using ProductElem = std::variant<int64_t, std::vector<int64_t>>;
    std::vector<std::vector<ProductElem>> all_lists = {
        // BlockConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.m_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.n_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.block_.k_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.m_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.n_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_.k_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpTileConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.m_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.n_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile_config_.warp_tile_.k_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding_.m_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding_.n_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding_.k_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.launch_.min_block_per_cu_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PartitionConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.partition_.num_wave_groups_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.partition_.tile_partitioner_group_num_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.partition_.tile_partitioner_m01_.values_)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PipelineConfig (enum as int64_t)
        [&]{ std::vector<ProductElem> v; for (const auto& x : flatten(config.pipeline_.version_.values_)) v.emplace_back(static_cast<int64_t>(GetPipelineVersionEnumFromString(x))); return v; }(),
        [&]{ std::vector<ProductElem> v; for (const auto& x : flatten(config.pipeline_.scheduler_.values_)) v.emplace_back(static_cast<int64_t>(GetPipelineSchedulerEnumFromString(x))); return v; }(),
        // EpilogueConfig (enum as int64_t)
        [&]{ std::vector<ProductElem> v; for (const auto& x : flatten(config.epilogue_.values_)) v.emplace_back(static_cast<int64_t>(GetEpilogueEnumFromString(x))); return v; }(),
    };

    // All config fields are converted to int64_t (enums or bools) before the product
    CartesianProduct(all_lists, [&](const std::vector<ProductElem>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t m_block = std::get<int64_t>(vals[idx++]);
        int64_t n_block = std::get<int64_t>(vals[idx++]);
        int64_t k_block = std::get<int64_t>(vals[idx++]);
        // WarpConfig
        int64_t m_warp = std::get<int64_t>(vals[idx++]);
        int64_t n_warp = std::get<int64_t>(vals[idx++]);
        int64_t k_warp = std::get<int64_t>(vals[idx++]);
        // WarpTileConfig
        int64_t m_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t n_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t k_warp_tile = std::get<int64_t>(vals[idx++]);
        // PaddingConfig
        bool pad_m = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool pad_n = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool pad_k = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        // LaunchConfig
        int64_t min_block_per_cu = std::get<int64_t>(vals[idx++]);
        // PartitionConfig
        int64_t num_wave_groups = std::get<int64_t>(vals[idx++]);
        int64_t tile_partitioner_group_num = std::get<int64_t>(vals[idx++]);
        int64_t tile_partitioner_m01 = std::get<int64_t>(vals[idx++]);
        // PipelineConfig
        auto version = static_cast<PipelineVersionEnum>(std::get<int64_t>(vals[idx++]));
        auto scheduler = static_cast<PipelineSchedulerEnum>(std::get<int64_t>(vals[idx++]));
        // EpilogueConfig
        auto epilogue = static_cast<EpilogueEnum>(std::get<int64_t>(vals[idx++]));

        // Construct  GemmCodeGen
        GemmCodeGen gemm;
        gemm.problem_ = gemm_problem;
        gemm.tile_desc_ = GemmTileDesc{m_block, n_block, k_block, m_warp, n_warp, k_warp, m_warp_tile, n_warp_tile, k_warp_tile};
        gemm.pipeline_version_ = version;
        gemm.pipeline_scheduler_ = scheduler;
        gemm.epilogue_ = epilogue;
        gemm.is_pad_m_ = pad_m;
        gemm.is_pad_n_ = pad_n;
        gemm.is_pad_k_ = pad_k;
        gemm.min_block_per_cu_ = min_block_per_cu;
        gemm.num_wave_groups_ = num_wave_groups;
        gemm.tile_partitioner_group_num_ = tile_partitioner_group_num;
        gemm.tile_partitioner_m01_ = tile_partitioner_m01;
        result.push_back(gemm);
    });
    return result;
}

} // namespace tile
} // namespace flashck