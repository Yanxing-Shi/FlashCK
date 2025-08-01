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
    GemmBlockConfig block_tile;
    GemmWarpConfig block_warps;
    GemmWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmTileConfig, block_tile, block_warps, warp_tile)

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
    GemmTileConfig tile_shape;
    GemmPaddingConfig padding;
    GemmLaunchConfig launch;
    GemmPartitionConfig partition;
    GemmPipelineConfig pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmConfig, tile_shape, padding, launch, partition, pipeline)

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
    FmhaFwdBlockConfig block_tile;
    FmhaFwdWarpConfig block_warps;
    FmhaFwdWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdConfig {
    FmhaFwdTileConfig tile_shape;
    FmhaFwdPaddingConfig padding;
    FmhaFwdLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdConfig, tile_shape, padding, launch, pipeline)

// ========== FMHA Fwd append kv Structs ==========
struct FmhaFwdAppendKVBlockConfig {
    IntEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVBlockConfig, s, sk, d, dv)

struct FmhaFwdAppendKVTileConfig {
    FmhaFwdAppendKVBlockConfig block_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVTileConfig, block_tile)

struct FmhaFwdAppendKVPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVPaddingConfig, s, sk, d, dv)

struct FmhaFwdAppendKVLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVLaunchConfig, min_block_per_cu)

struct FmhaFwdAppendKVConfig {
    FmhaFwdAppendKVTileConfig tile_shape;
    FmhaFwdAppendKVPaddingConfig padding;
    FmhaFwdAppendKVLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVConfig, tile_shape, padding, launch)

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
    FmhaFwdSplitKVBlockConfig block_tile;
    FmhaFwdSplitKVWarpConfig block_warps;
    FmhaFwdSplitKVWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdSplitKVPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVPaddingConfig, s, sk, d, dv)

struct FmhaFwdSplitKVLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVLaunchConfig, min_block_per_cu)

struct FmhaFwdSplitKVConfig {
    FmhaFwdSplitKVTileConfig tile_shape;
    FmhaFwdSplitKVPaddingConfig padding;
    FmhaFwdSplitKVLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVConfig, tile_shape, padding, launch, pipeline)

// ========== FMHA Fwd split kv combine Structs ==========
struct FmhaFwdSplitKVCombineBlockConfig {
    IntEnumConfigParam n1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineBlockConfig, n1)

struct FmhaFwdSplitKVCombineTileConfig {
    FmhaFwdSplitKVCombineBlockConfig block_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineTileConfig, block_tile)

struct FmhaFwdSplitKVCombinePaddingConfig {
    BoolEnumConfigParam s, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombinePaddingConfig, s, dv)

struct FmhaFwdSplitKVCombineLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineLaunchConfig, min_block_per_cu)

struct FmhaFwdSplitKVCombineConfig {
    FmhaFwdSplitKVCombineTileConfig tile_shape;
    FmhaFwdSplitKVCombinePaddingConfig padding;
    FmhaFwdSplitKVCombineLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineConfig, tile_shape, padding, launch)

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
    FmhaBatchPrefillBlockConfig block_tile;
    FmhaBatchPrefillWarpConfig block_warps;
    FmhaBatchPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillTileConfig, block_tile, block_warps, warp_tile)

struct FmhaBatchPrefillConfig {
    FmhaBatchPrefillTileConfig tile_shape;
    FmhaBatchPrefillPaddingConfig padding;
    FmhaBatchPrefillLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaBatchPrefillConfig, tile_shape, padding, launch, pipeline)

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
    FmhaPagedKVPrefillBlockConfig block_tile;
    FmhaPagedKVPrefillWarpConfig block_warps;
    FmhaPagedKVPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillTileConfig, block_tile, block_warps, warp_tile)

struct FmhaPagedKVPrefillConfig {
    FmhaPagedKVPrefillTileConfig tile_shape;
    FmhaPagedKVPrefillPaddingConfig padding;
    FmhaPagedKVPrefillLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaPagedKVPrefillConfig, tile_shape, padding, launch, pipeline)

// ========== Norm Structs ==========
struct NormTileConfig {
    IntEnumConfigParam m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormTileConfig, m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector)

struct NormPaddingConfig {
    BoolEnumConfigParam n;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormPaddingConfig, n)

struct NormLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormLaunchConfig, min_block_per_cu)

struct NormConfig {
    NormTileConfig tile_shape;
    NormPaddingConfig padding;
    NormLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormConfig, tile_shape, padding, launch)

// ========== Moe Structs ==========
struct MoeBlockConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeBlockConfig, m, n, k)

struct MoeWarpConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeWarpConfig, m, n, k)

struct MoeWarpTileConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeWarpTileConfig, m, n, k)

struct MoeTileConfig {
    MoeBlockConfig block_tile;
    MoeWarpConfig block_warps;
    MoeWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeTileConfig, block_tile, block_warps, warp_tile)

struct MoeLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeLaunchConfig, min_block_per_cu)
struct MoeConfig {
    MoeTileConfig tile_shape;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeConfig, tile_shape, launch);

struct MoeSortingConfig {
    IntEnumConfigParam internal_load_unroll;
    IntEnumConfigParam expert_tile;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSortingConfig, internal_load_unroll, expert_tile, launch);

struct TopkSoftmaxConfig {
    IntEnumConfigParam issue_per_col;
    IntEnumConfigParam launch_type;
    IntEnumConfigParam bytes_per_issue;
    IntEnumConfigParam block_size;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TopkSoftmaxConfig, issue_per_col, launch_type, bytes_per_issue, block_size, launch);

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
