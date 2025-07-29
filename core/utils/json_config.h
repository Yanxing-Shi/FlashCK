#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

namespace flashck {

using json = nlohmann::json;

// ========== Common Parameter Structs ==========

struct IntEnumConfigParam {
    std::vector<std::vector<int>> values;
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
    j = nlohmann::json{{"values", p.values}};
}

struct BoolEnumConfigParam {
    std::vector<bool> values;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BoolEnumConfigParam, values)

struct StrEnumConfigParam {
    std::vector<std::string> values;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(StrEnumConfigParam, values)

// ========== Legacy GEMM Config Structs ==========
struct LegacyTileConfig {
    IntEnumConfigParam scale_block_size, block_size, m_per_block, n_per_block, k_per_block, a_k1, b_k1, m_per_xdl, n_per_xdl, m_xdl_per_wave, n_xdl_per_wave;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyTileConfig, scale_block_size, block_size, m_per_block, n_per_block, k_per_block, a_k1, b_k1, m_per_xdl, n_per_xdl, m_xdl_per_wave, n_xdl_per_wave)

struct LegacyBlockTransferConfig {
    IntEnumConfigParam thread_cluster_length;
    IntEnumConfigParam arrange_order;
    IntEnumConfigParam src_access_order;
    IntEnumConfigParam src_vector_dim;
    IntEnumConfigParam src_scalar_per_vector;
    IntEnumConfigParam dst_scalar_per_vector_k1;
    IntEnumConfigParam lds_add_extra_m;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyBlockTransferConfig, thread_cluster_length, arrange_order, src_access_order, src_vector_dim, src_scalar_per_vector, dst_scalar_per_vector_k1, lds_add_extra_m)

struct LegacyCBlockTransferConfig {
    IntEnumConfigParam m_xdl_per_wave, n_xdl_per_wave, thread_cluster_length, scalar_per_vector;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyCBlockTransferConfig, m_xdl_per_wave, n_xdl_per_wave, thread_cluster_length, scalar_per_vector);

struct PipelineConfig{
    StrEnumConfigParam version, scheduler;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PipelineConfig, version, scheduler)

struct LegacyGemmConfig {
    LegacyTileConfig tile;
    LegacyBlockTransferConfig a_block;
    LegacyBlockTransferConfig b_block;
    LegacyCBlockTransferConfig c_block;
    PipelineConfig pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LegacyGemmConfig, tile, a_block, b_block, c_block, pipeline)

// ========== Tile-based GEMM Structs ==========
struct BlockConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BlockConfig, m, n, k)

struct WarpConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WarpConfig, m, n, k)

struct WarpTileConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WarpTileConfig, m, n, k)

struct TileConfig {
    BlockConfig block;
    WarpConfig warp;
    WarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileConfig, block, warp, warp_tile)

struct PaddingConfig {
    BoolEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PaddingConfig, m, n, k)

struct LaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LaunchConfig, min_block_per_cu)

struct PartitionConfig {
    IntEnumConfigParam num_wave_groups, tile_partitioner_group_num, tile_partitioner_m01;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PartitionConfig, num_wave_groups, tile_partitioner_group_num, tile_partitioner_m01)
struct TileGemmConfig {
    TileConfig tile_config;
    PaddingConfig padding;
    LaunchConfig launch;
    PartitionConfig partition;
    PipelineConfig pipeline;
    StrEnumConfigParam epilogue;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileGemmConfig, tile_config, padding, launch, partition, pipeline, epilogue)

// ========== FMHA Fwd Structs ==========
struct FmhaFwdBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPaddingConfig, s, sk, d, dv)

struct FmhaFwdLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdLaunchConfig, min_block_per_cu)

struct FmhaFwdTileConfig {
    FmhaFwdBlockConfig block;
    FmhaFwdWarpConfig warp;
    FmhaFwdWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdTileConfig, block, warp, warp_tile)

struct FmhaFwdConfig {
    FmhaFwdTileConfig tile;
    FmhaFwdPaddingConfig padding;
    FmhaFwdLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdConfig, tile, padding, launch, pipeline)

// ========== FMHA Fwd append kv Structs ==========
struct FmhaFwdAppendKVBlockConfig {
    IntEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVBlockConfig, s, sk, d, dv)

struct FmhaFwdAppendKVTileConfig {
    FmhaFwdAppendKVBlockConfig block;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVTileConfig, block)

struct FmhaFwdAppendKVConfig {
    FmhaFwdTileConfig tile;
    FmhaFwdPaddingConfig padding;
    FmhaFwdLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVConfig, tile, padding, launch)

// ========== FMHA Fwd split kv Structs ==========

struct FmhaFwdSplitKVTileConfig {
    FmhaFwdBlockConfig block;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVTileConfig, block)

struct FmhaFwdSplitKVConfig {
    FmhaFwdSplitKVTileConfig tile;
    FmhaFwdPaddingConfig padding;
    FmhaFwdLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVConfig, tile, padding, launch)

// ========== FMHA Fwd split kv combine Structs ==========

struct FmhaFwdSplitKVCombineBlockConfig {
    FmhaFwdBlockConfig m0, n1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineBlockConfig, m0, n1)

struct FmhaFwdSplitKVCombineTileConfig {
    FmhaFwdBlockConfig block;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineTileConfig, block)

struct FmhaFwdSplitKVCombinePaddingConfig {
    BoolEnumConfigParam s, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombinePaddingConfig, s, dv)

struct FmhaFwdSplitKVCombineConfig {
    FmhaFwdSplitKVCombineTileConfig tile;
    FmhaFwdSplitKVCombinePaddingConfig padding;
    FmhaFwdLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineConfig, tile, padding, launch)

// ========== FMHA batch prefill Structs ==========
struct FmhaBatchPrefillConfig {
    FmhaFwdTileConfig tile;
    FmhaFwdPaddingConfig padding;
    FmhaFwdLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillConfig, tile, padding, launch, pipeline)

// ========== FMHA bat Structs ==========
struct FmhaPagedKVConfig {
    FmhaFwdTileConfig tile;
    FmhaFwdPaddingConfig padding;
    FmhaFwdLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVConfig, tile, padding, launch, pipeline)


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
