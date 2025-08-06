#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace flashck {

using json = nlohmann::json;

// ========== Common Parameter Structs ==========

struct IntEnumConfigParam {
    json data;  // Store raw JSON data to support both modes
    
    // Get all values, supporting both "values" and "min/max/step/exclude" modes
    std::vector<int> GetAllValues() const {
        if (data.contains("values") && data["values"].is_array()) {
            return data["values"].get<std::vector<int>>();
        }
        
        // Generate from min/max/step/exclude
        std::vector<int> result;
        if (data.contains("min") && data.contains("max")) {
            int min_val = data["min"].get<int>();
            int max_val = data["max"].get<int>();
            int step_val = data.contains("step") ? data["step"].get<int>() : 1;
            std::vector<int> exclude_vals;
            if (data.contains("exclude")) {
                exclude_vals = data["exclude"].get<std::vector<int>>();
            }
            
            for (int v = min_val; v <= max_val; v += step_val) {
                if (std::find(exclude_vals.begin(), exclude_vals.end(), v) == exclude_vals.end()) {
                    result.push_back(v);
                }
            }
        }
        return result;
    }
    
    // Legacy compatibility - access values directly
    std::vector<int> values;
    
    // Constructor to maintain backward compatibility
    IntEnumConfigParam() = default;
    IntEnumConfigParam(const std::vector<int>& vals) : values(vals) {
        data["values"] = vals;
    }
};

// Custom JSON serialization for IntEnumConfigParam
inline void to_json(json& j, const IntEnumConfigParam& p) {
    j = p.data;
}

inline void from_json(const json& j, IntEnumConfigParam& p) {
    p.data = j;
    // For backward compatibility, populate values if present
    if (j.contains("values")) {
        p.GetAllValues() = j["values"].get<std::vector<int>>();
    } else {
        p.GetAllValues() = p.GetAllValues();
    }
}

struct BoolEnumConfigParam {
    json data;  // Store raw JSON data to support both modes
    
    // Get all values, supporting both "values" and "min/max/exclude" modes
    std::vector<bool> GetAllValues() const {
        if (data.contains("values") && data["values"].is_array()) {
            return data["values"].get<std::vector<bool>>();
        }
        
        // Generate from min/max/exclude (for bool, typically just [false, true])
        std::vector<bool> result;
        if (data.contains("min") && data.contains("max")) {
            bool min_val = data["min"].get<bool>();
            bool max_val = data["max"].get<bool>();
            std::vector<bool> exclude_vals;
            if (data.contains("exclude")) {
                exclude_vals = data["exclude"].get<std::vector<bool>>();
            }
            
            std::vector<bool> candidates = {false, true};
            for (bool v : candidates) {
                if (v >= min_val && v <= max_val) {
                    if (std::find(exclude_vals.begin(), exclude_vals.end(), v) == exclude_vals.end()) {
                        result.push_back(v);
                    }
                }
            }
        }
        return result;
    }
    
    // Legacy compatibility - access values directly
    std::vector<bool> values;
    
    // Constructor to maintain backward compatibility
    BoolEnumConfigParam() = default;
    BoolEnumConfigParam(const std::vector<bool>& vals) : values(vals) {
        data["values"] = vals;
    }
};

// Custom JSON serialization for BoolEnumConfigParam
inline void to_json(json& j, const BoolEnumConfigParam& p) {
    j = p.data;
}

inline void from_json(const json& j, BoolEnumConfigParam& p) {
    p.data = j;
    // For backward compatibility, populate values if present
    if (j.contains("values")) {
        p.GetAllValues() = j["values"].get<std::vector<bool>>();
    } else {
        p.GetAllValues() = p.GetAllValues();
    }
}

struct StrEnumConfigParam {
    json data;  // Store raw JSON data to support both modes
    
    // Get all values, supporting both "values" and "exclude" modes
    std::vector<std::string> GetAllValues() const {
        if (data.contains("values") && data["values"].is_array()) {
            return data["values"].get<std::vector<std::string>>();
        }
        
        // For strings, min/max/step typically not meaningful, so just return empty
        // Could be extended if needed for specific use cases
        return {};
    }
    
    // Legacy compatibility - access values directly
    std::vector<std::string> values;
    
    // Constructor to maintain backward compatibility
    StrEnumConfigParam() = default;
    StrEnumConfigParam(const std::vector<std::string>& vals) : values(vals) {
        data["values"] = vals;
    }
};

// Custom JSON serialization for StrEnumConfigParam
inline void to_json(json& j, const StrEnumConfigParam& p) {
    j = p.data;
}

inline void from_json(const json& j, StrEnumConfigParam& p) {
    p.data = j;
    // For backward compatibility, populate values if present
    if (j.contains("values")) {
        p.GetAllValues() = j["values"].get<std::vector<std::string>>();
    } else {
        p.GetAllValues() = p.GetAllValues();
    }
}


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
    BoolEnumConfigParam skip_min_q_seq_len;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdConfig, tile_shape, padding, launch, skip_min_q_seq_len, pipeline)

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
    IntEnumConfigParam num_splits;
    BoolEnumConfigParam has_uneven_splits;
    BoolEnumConfigParam merge_groups_num_head_q_seq_len;
    FmhaFwdSplitKVLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVConfig, tile_shape, padding, num_splits, launch, has_uneven_splits, merge_groups_num_head_q_seq_len, pipeline)

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
struct FmhaFwdBatchPrefillBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdBatchPrefillWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdBatchPrefillWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdBatchPrefillPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillPaddingConfig, s, sk, d, dv)

struct FmhaFwdBatchPrefillLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillLaunchConfig, min_block_per_cu)

struct FmhaFwdBatchPrefillTileConfig {
    FmhaFwdBatchPrefillBlockConfig block_tile;
    FmhaFwdBatchPrefillWarpConfig block_warps;
    FmhaFwdBatchPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdBatchPrefillConfig {
    FmhaFwdBatchPrefillTileConfig tile_shape;
    FmhaFwdBatchPrefillPaddingConfig padding;
    FmhaFwdBatchPrefillLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillConfig, tile_shape, padding, launch, pipeline)

// ========== FMHA Structs ==========
struct FmhaFwdPagedKVPrefillBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdPagedKVPrefillWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdPagedKVPrefillWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdPagedKVPrefillPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillPaddingConfig, s, sk, d, dv)

struct FmhaFwdPagedKVPrefillLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillLaunchConfig, min_block_per_cu)

struct FmhaFwdPagedKVPrefillTileConfig {
    FmhaFwdPagedKVPrefillBlockConfig block_tile;
    FmhaFwdPagedKVPrefillWarpConfig block_warps;
    FmhaFwdPagedKVPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdPagedKVPrefillConfig {
    FmhaFwdPagedKVPrefillTileConfig tile_shape;
    FmhaFwdPagedKVPrefillPaddingConfig padding;
    FmhaFwdPagedKVPrefillLaunchConfig launch;
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillConfig, tile_shape, padding, launch, pipeline)

// ========== Norm Structs ==========
struct NormTileConfig {
    IntEnumConfigParam m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormTileConfig, m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector)

struct NormPaddingConfig {
    BoolEnumConfigParam n;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormPaddingConfig, n)

struct NormPipelineConfig {
    BoolEnumConfigParam is_two_pass;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormPipelineConfig, is_two_pass)

struct NormLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormLaunchConfig, min_block_per_cu)

struct NormConfig {
    NormTileConfig tile_shape;
    NormPaddingConfig padding;
    NormPipelineConfig pipeline;
    NormLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormConfig, tile_shape, padding, pipeline, launch)

// ========== Moe Structs ==========
struct MoeGemmBlockConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmBlockConfig, m, n, k)

struct MoeGemmWarpConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmWarpConfig, m, n, k)

struct MoeGemmWarpTileConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmWarpTileConfig, m, n, k)

struct MoeGemmTileConfig {
    MoeGemmBlockConfig block_tile;
    MoeGemmWarpConfig block_warps;
    MoeGemmWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmTileConfig, block_tile, block_warps, warp_tile)

struct MoeGemmPipelineConfig {
    BoolEnumConfigParam interleave;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmPipelineConfig, interleave)

struct MoeGemmPaddingConfig {
    BoolEnumConfigParam hidden_size;
    BoolEnumConfigParam intermediate_size;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmPaddingConfig, hidden_size, intermediate_size)


struct MoeLaunchConfig {
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeLaunchConfig, min_block_per_cu)

struct MoeGemmConfig {
    MoeGemmTileConfig tile_shape;
    MoeGemmPaddingConfig padding;
    MoeGemmPipelineConfig pipeline;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmConfig, tile_shape, padding, pipeline, launch);

struct MoeSmoothQuantTileConfig {
    IntEnumConfigParam m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantTileConfig, m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector)

struct MoeSmoothQuantPaddingConfig {
    BoolEnumConfigParam n;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantPaddingConfig, n)

struct MoeSmoothQuantPipelineConfig {
    BoolEnumConfigParam is_two_pass;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantPipelineConfig, is_two_pass)

struct MoeSmoothQuantConfig {
    MoeSmoothQuantTileConfig tile_shape;
    MoeSmoothQuantPaddingConfig padding;
    MoeSmoothQuantPipelineConfig pipeline;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantConfig, tile_shape, padding, pipeline, launch)

struct MoeSortingConfig {
    IntEnumConfigParam internal_load_unroll;
    IntEnumConfigParam expert_tile;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSortingConfig, internal_load_unroll, expert_tile, launch);

struct TopKSoftmaxConfig {
    IntEnumConfigParam issues_per_col;
    IntEnumConfigParam launch_type;
    IntEnumConfigParam bytes_per_issue;
    IntEnumConfigParam block_size;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TopKSoftmaxConfig, issues_per_col, launch_type, bytes_per_issue, block_size, launch);

// ========== Config Loader ==========
// Generic loader for config (single or array)
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
