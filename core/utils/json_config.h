#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

namespace flashck {

using json = nlohmann::json;

// ========== Common Parameter Structs ==========

struct IntEnumConfigParam {
    std::vector<std::vector<int>> values_;
};

// Custom serialization to support both vector<int> and vector<vector<int>>
inline void from_json(const nlohmann::json& j, IntEnumConfigParam& p) {
    if (j.at("values_").is_array() && !j.at("values_").empty() && j.at("values_")[0].is_array()) {
        j.at("values_").get_to(p.values_);
    } else {
        std::vector<int> tmp;
        j.at("values_").get_to(tmp);
        p.values_ = {tmp};
    }
}

inline void to_json(nlohmann::json& j, const IntEnumConfigParam& p) {
    j = nlohmann::json{{"values_", p.values_}};
}

struct BoolEnumConfigParam {
    std::vector<bool> values_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BoolEnumConfigParam, values_)

struct StrEnumConfigParam {
    std::vector<std::string> values_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(StrEnumConfigParam, values_)

// ========== Legacy GEMM Config Structs ==========
struct LegacyTileConfig {
    IntEnumConfigParam scale_block_size_, block_size_, m_per_block_, n_per_block_, k_per_block_, a_k1_, b_k1_, m_per_xdl_, n_per_xdl_, m_xdl_per_wave_, n_xdl_per_wave_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyTileConfig, scale_block_size_, block_size_, m_per_block_, n_per_block_, k_per_block_, a_k1_, b_k1_, m_per_xdl_, n_per_xdl_, m_xdl_per_wave_, n_xdl_per_wave_)

struct LegacyBlockTransferConfig {
    IntEnumConfigParam thread_cluster_length_;
    IntEnumConfigParam arrange_order_;
    IntEnumConfigParam src_access_order_;
    IntEnumConfigParam src_vector_dim_;
    IntEnumConfigParam src_scalar_per_vector_;
    IntEnumConfigParam dst_scalar_per_vector_k1_;
    IntEnumConfigParam lds_add_extra_m_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyBlockTransferConfig, thread_cluster_length_, arrange_order_, src_access_order_, src_vector_dim_, src_scalar_per_vector_, dst_scalar_per_vector_k1_, lds_add_extra_m_)

struct LegacyCBlockTransferConfig {
    IntEnumConfigParam m_xdl_per_wave_, n_xdl_per_wave_, thread_cluster_length_, scalar_per_vector_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyCBlockTransferConfig, m_xdl_per_wave_, n_xdl_per_wave_, thread_cluster_length_, scalar_per_vector_)

struct PipelineConfig{
    StrEnumConfigParam version_, scheduler_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PipelineConfig, version_, scheduler_)

struct LegacyGemmConfig {
    LegacyTileConfig tile_config_;
    LegacyBlockTransferConfig a_block_config_;
    LegacyBlockTransferConfig b_block_config_;
    LegacyCBlockTransferConfig c_block_config_;
    PipelineConfig pipeline_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyGemmConfig, tile_config_, a_block_config_, b_block_config_, c_block_config_, pipeline_)

// ========== Tile-based GEMM Structs ==========
struct BlockConfig {
    IntEnumConfigParam m_, n_, k_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BlockConfig, m_, n_, k_)

struct WarpConfig {
    IntEnumConfigParam m_, n_, k_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WarpConfig, m_, n_, k_)

struct WarpTileConfig {
    IntEnumConfigParam m_, n_, k_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WarpTileConfig, m_, n_, k_)

struct TileConfig {
    BlockConfig block_;
    WarpConfig warp_;
    WarpTileConfig warp_tile_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileConfig, block_, warp_, warp_tile_)

struct PaddingConfig {
    BoolEnumConfigParam m_, n_, k_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PaddingConfig, m_, n_, k_)

struct LaunchConfig {
    IntEnumConfigParam min_block_per_cu_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LaunchConfig, min_block_per_cu_)

struct PartitionConfig {
    IntEnumConfigParam num_wave_groups_, tile_partitioner_group_num_, tile_partitioner_m01_;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PartitionConfig, num_wave_groups_, tile_partitioner_group_num_, tile_partitioner_m01_)
struct TileGemmConfig {
    TileConfig tile_config_;
    PaddingConfig padding_;
    LaunchConfig launch_;
    PartitionConfig partition_;
    PipelineConfig pipeline_;
    StrEnumConfigParam epilogue_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileGemmConfig, tile_config_, padding_, launch_, partition_, pipeline_, epilogue_)

// ========== FMHA Fwd Structs ==========
struct FmhaFwdBlockConfig {
    IntEnumConfigParam m0_, n0_, k0_, n1_, k1_, k0_max_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBlockConfig, m0_, n0_, k0_, n1_, k1_, k0_max_)

struct FmhaFwdWarpConfig {
    IntEnumConfigParam m0_, n0_, k0_, m1_, n1_, k1_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdWarpConfig, m0_, n0_, k0_, m1_, n1_, k1_)

struct FmhaFwdWarpTileConfig {
    IntEnumConfigParam m0_, n0_, k0_, m1_, n1_, k1_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdWarpTileConfig, m0_, n0_, k0_, m1_, n1_, k1_)

struct FmhaFwdPaddingConfig {
    BoolEnumConfigParam s_, sk_, d_, dv_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPaddingConfig, s_, sk_, d_, dv_)

struct FmhaFwdLaunchConfig {
    IntEnumConfigParam min_block_per_cu_, max_thread_per_block_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdLaunchConfig, min_block_per_cu_, max_thread_per_block_)

struct FmhaFwdTileConfig {
    FmhaFwdBlockConfig block_;
    FmhaFwdWarpConfig warp_;
    FmhaFwdWarpTileConfig warp_tile_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdTileConfig, block_, warp_, warp_tile_)

struct FmhaFwdConfig {
    FmhaFwdTileConfig tile_config_;
    FmhaFwdPaddingConfig padding_;
    FmhaFwdLaunchConfig launch_;
    StrEnumConfigParam pipeline_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdConfig, tile_config_, padding_, launch_, pipeline_)

// ========== FMHA Fwd append kv Structs ==========
struct FmhaFwdAppendKVBlockConfig {
    IntEnumConfigParam s_, sk_ , d_, dv_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVBlockConfig, s_, sk_ , d_, dv_)

struct FmhaFwdAppendKVTileConfig {
    FmhaFwdAppendKVBlockConfig block_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVTileConfig, block_)

struct FmhaFwdAppendKVConfig {
    FmhaFwdTileConfig tile_config_;
    FmhaFwdPaddingConfig padding_;
    FmhaFwdLaunchConfig launch_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVConfig, tile_config_, padding_, launch_)

// ========== FMHA Fwd split kv Structs ==========

struct FmhaFwdSplitKVTileConfig {
    FmhaFwdBlockConfig block_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVTileConfig, block_)

struct FmhaFwdSplitKVConfig {
    FmhaFwdSplitKVTileConfig tile_config_;
    FmhaFwdPaddingConfig padding_;
    FmhaFwdLaunchConfig launch_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVConfig, tile_config_, padding_, launch_)

// ========== FMHA Fwd split kv combine Structs ==========

struct FmhaFwdSplitKVCombineBlockConfig {
    FmhaFwdBlockConfig m0_, n1_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineBlockConfig, m0_, n1_)

struct FmhaFwdSplitKVCombineTileConfig {
    FmhaFwdBlockConfig block_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineTileConfig, block_)

struct FmhaFwdSplitKVCombinePaddingConfig {
    BoolEnumConfigParam s_, dv_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombinePaddingConfig, s_, dv_)

struct FmhaFwdSplitKVCombineConfig {
    FmhaFwdSplitKVCombineTileConfig tile_config_;
    FmhaFwdSplitKVCombinePaddingConfig padding_;
    FmhaFwdLaunchConfig launch_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineConfig, tile_config_, padding_, launch_)

// ========== FMHA batch prefill Structs ==========
struct FmhaBatchPrefillConfig {
    FmhaFwdTileConfig tile_config_;
    FmhaFwdPaddingConfig padding_;
    FmhaFwdLaunchConfig launch_;
    StrEnumConfigParam pipeline_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillConfig, tile_config_, padding_, launch_, pipeline_)

// ========== FMHA bat Structs ==========
struct FmhaPagedKVConfig {
    FmhaFwdTileConfig tile_config_;
    FmhaFwdPaddingConfig padding_;
    FmhaFwdLaunchConfig launch_;
    StrEnumConfigParam pipeline_;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVConfig, tile_config_, padding_, launch_, pipeline_)


// Generic loader for any config type
template <typename T>
inline T LoadConfigJson(const std::string& path) {
    std::ifstream f(path);
    json j;
    f >> j;
    return j.get<T>();
}

using ProductElem = std::variant<int64_t, std::vector<int64_t>>;

inline void CartesianProduct(
    const std::vector<std::vector<ProductElem>>& value_lists,
    std::function<void(const std::vector<ProductElem>&)> callback)
{
    std::vector<ProductElem> current;
    std::function<void(size_t)> recurse = [&](size_t depth) {
        if (depth == value_lists.size()) {
            callback(current);
            return;
        }
        for (const auto& v : value_lists[depth]) {
            current.push_back(v);
            recurse(depth + 1);
            current.pop_back();
        }
    };
    recurse(0);
}

} // namespace flashck
