#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

namespace flashck {

using json = nlohmann::json;

// ========== Common Parameter Structs ==========

struct IntEnumConfigParam {
    std::vector<int> values;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IntEnumConfigParam, values)

struct BoolEnumConfigParam {
    std::vector<bool> values;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BoolEnumConfigParam, values)

struct StrEnumConfigParam {
    std::vector<std::string> values;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(StrEnumConfigParam, values)


// ========== Tile-based GEMM Structs ==========
struct GemmBlockConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmBlockConfig, m, n, k)

struct GemmWarpConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmWarpConfig, m, n, k)

struct GemmWarpTileConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmWarpTileConfig, m, n, k)

struct GemmTileConfig {
    GemmBlockConfig block;
    GemmWarpConfig warp;
    GemmWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmTileConfig, block, warp, warp_tile)

struct GemmPaddingConfig {
    BoolEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmPaddingConfig, m, n, k)

struct GemmLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmLaunchConfig, min_block_per_cu)

struct GemmPartitionConfig {
    IntEnumConfigParam num_wave_groups, tile_partitioner_group_num, tile_partitioner_m01;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmPartitionConfig, num_wave_groups, tile_partitioner_group_num, tile_partitioner_m01)

struct GemmPipelineConfig{
    StrEnumConfigParam version, scheduler, epilogue;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmPipelineConfig, version, scheduler, epilogue)

struct GemmConfig {
    GemmTileConfig tile;
    GemmPaddingConfig padding;
    GemmLaunchConfig launch;
    GemmPartitionConfig partition;
    GemmPipelineConfig pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmConfig, tile, padding, launch, partition, pipeline)

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

struct FmhaFwdAppendKVPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVPaddingConfig, s, sk, d, dv)

struct FmhaFwdAppendKVLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVLaunchConfig, min_block_per_cu)

struct FmhaFwdAppendKVConfig {
    FmhaFwdAppendKVTileConfig tile;
    FmhaFwdAppendKVPaddingConfig padding;
    FmhaFwdAppendKVLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVConfig, tile, padding, launch)

// ========== FMHA Fwd split kv Structs ==========

struct FmhaFwdSplitKVBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdSplitKVWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdSplitKVWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVWarpTileConfig, m0, n0, k0, m1, n1, k1)
struct FmhaFwdSplitKVTileConfig {
    FmhaFwdSplitKVBlockConfig block;
    FmhaFwdSplitKVWarpConfig warp;
    FmhaFwdSplitKVWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVTileConfig, block, warp, warp_tile)

struct FmhaFwdSplitKVPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVPaddingConfig, s, sk, d, dv)

struct FmhaFwdSplitKVLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVLaunchConfig, min_block_per_cu)

struct FmhaFwdSplitKVConfig {
    FmhaFwdSplitKVTileConfig tile;
    FmhaFwdSplitKVPaddingConfig padding;
    FmhaFwdSplitKVLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVConfig, tile, padding, launch, pipeline)

// ========== FMHA Fwd split kv combine Structs ==========
struct FmhaFwdSplitKVCombineBlockConfig {
    IntEnumConfigParam n1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineBlockConfig, n1)

struct FmhaFwdSplitKVCombineTileConfig {
    FmhaFwdSplitKVCombineBlockConfig block;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineTileConfig, block)

struct FmhaFwdSplitKVCombinePaddingConfig {
    BoolEnumConfigParam s, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombinePaddingConfig, s, dv)

struct FmhaFwdSplitKVCombineLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineLaunchConfig, min_block_per_cu)

struct FmhaFwdSplitKVCombineConfig {
    FmhaFwdSplitKVCombineTileConfig tile;
    FmhaFwdSplitKVCombinePaddingConfig padding;
    FmhaFwdSplitKVCombineLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineConfig, tile, padding, launch)

// ========== FMHA batch prefill Structs ==========
struct FmhaBatchPrefillBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaBatchPrefillWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaBatchPrefillWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaBatchPrefillPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillPaddingConfig, s, sk, d, dv)

struct FmhaBatchPrefillLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillLaunchConfig, min_block_per_cu)

struct FmhaBatchPrefillTileConfig {
    FmhaBatchPrefillBlockConfig block;
    FmhaBatchPrefillWarpConfig warp;
    FmhaBatchPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillTileConfig, block, warp, warp_tile)

struct FmhaBatchPrefillConfig {
    FmhaBatchPrefillTileConfig tile;
    FmhaBatchPrefillPaddingConfig padding;
    FmhaBatchPrefillLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillConfig, tile, padding, launch, pipeline)

// ========== FMHA Structs ==========
struct FmhaPagedKVPrefillBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaPagedKVPrefillWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaPagedKVPrefillWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaPagedKVPrefillPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillPaddingConfig, s, sk, d, dv)

struct FmhaPagedKVPrefillLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillLaunchConfig, min_block_per_cu)

struct FmhaPagedKVPrefillTileConfig {
    FmhaPagedKVPrefillBlockConfig block;
    FmhaPagedKVPrefillWarpConfig warp;
    FmhaPagedKVPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillTileConfig, block, warp, warp_tile)

struct FmhaPagedKVPrefillConfig {
    FmhaPagedKVPrefillTileConfig tile;
    FmhaPagedKVPrefillPaddingConfig padding;
    FmhaPagedKVPrefillLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillConfig, tile, padding, launch, pipeline)

// ========== Config Loader ==========
// Generic loader for any config type
template <typename T>
inline T LoadConfigJson(const std::string& path) {
    std::ifstream f(path);
    json j;
    f >> j;
    return j.get<T>();
}

inline void CartesianProduct(
    const std::vector<std::vector<int64_t>>& value_lists,
    std::function<void(const std::vector<int64_t>&)> callback)
{
    std::vector<int64_t> current;
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
