#include "core/profiling/fmha/fmha_fwd_split_kv/fmha_fwd_split_kv_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);     // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);    // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);   // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);      // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);    // Base path for config files

namespace flashck {

bool FmhaFwdSplitKVEmitter::IsValidTile(const FmhaFwdSplitKVTileDesc& tile_desc, const FmhaProblem& fmha_problem)
{
    // Validate all tile parameters are positive
    if (tile_desc.m0_block_ <= 0 || tile_desc.n0_block_ <= 0 || tile_desc.k0_block_ <= 0 || 
        tile_desc.k0_max_block_ <= 0 || tile_desc.n1_block_ <= 0 || tile_desc.k1_block_ <= 0 ||
        tile_desc.m0_warp_ <= 0 || tile_desc.n0_warp_ <= 0 || tile_desc.k0_warp_ < 0 ||
        tile_desc.m1_warp_ <= 0 || tile_desc.n1_warp_ <= 0 || tile_desc.k1_warp_ < 0 ||
        tile_desc.m0_warp_tile_ <= 0 || tile_desc.n0_warp_tile_ <= 0 || tile_desc.k0_warp_tile_ <= 0 ||
        tile_desc.m1_warp_tile_ <= 0 || tile_desc.n1_warp_tile_ <= 0 || tile_desc.k1_warp_tile_ <= 0) {
        VLOG(3) << "Invalid FMHA split KV tile: negative or zero values not allowed";
        return false;
    }

    // Validate k0_block_ <= k0_max_block_
    if (tile_desc.k0_block_ > tile_desc.k0_max_block_) {
        VLOG(3) << "Invalid FMHA split KV tile: k0_block_ > k0_max_block_";
        return false;
    }

    // Validate warp*warp_tile <= block sizes
    if (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n0_warp_ * tile_desc.n0_warp_tile_ > tile_desc.n0_block_ ||
        tile_desc.k0_warp_ * tile_desc.k0_warp_tile_ > tile_desc.k0_block_ ||
        tile_desc.m1_warp_ * tile_desc.m1_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n1_warp_ * tile_desc.n1_warp_tile_ > tile_desc.n1_block_ ||
        tile_desc.k1_warp_ * tile_desc.k1_warp_tile_ > tile_desc.k1_block_) {
        VLOG(3) << "Invalid FMHA split KV tile: warp*warp_tile exceeds block size";
        return false;
    }

    // Validate block sizes are divisible by warp*warp_tile
    if ((tile_desc.m0_block_ % (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_) != 0) ||
        (tile_desc.n0_block_ % (tile_desc.n0_warp_ * tile_desc.n0_warp_tile_) != 0) ||
        (tile_desc.k0_block_ % (std::max<int64_t>(1, tile_desc.k0_warp_ * tile_desc.k0_warp_tile_)) != 0) ||
        (tile_desc.m0_block_ % (tile_desc.m1_warp_ * tile_desc.m1_warp_tile_) != 0) ||
        (tile_desc.n1_block_ % (tile_desc.n1_warp_ * tile_desc.n1_warp_tile_) != 0) ||
        (tile_desc.k1_block_ % (std::max<int64_t>(1, tile_desc.k1_warp_ * tile_desc.k1_warp_tile_)) != 0)) {
        VLOG(3) << "Invalid FMHA split KV tile: block size not divisible by warp*warp_tile";
        return false;
    }

    // Validate against problem dimensions for Batch mode
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.m0_block_ > fmha_problem.q_seq_len_ || tile_desc.n0_block_ > fmha_problem.kv_seq_len_ ||
            tile_desc.n1_block_ > fmha_problem.v_head_dim_ || tile_desc.k0_max_block_ > fmha_problem.qk_head_dim_) {
            VLOG(3) << "Invalid FMHA split KV tile: tile dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaFwdSplitKVEmitter::IsValidInstance(const FmhaFwdSplitKVCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
} 

std::vector<FmhaFwdSplitKVCodeGen> FmhaFwdSplitKVEmitter::HeuristicFilter(const std::vector<FmhaFwdSplitKVCodeGen>& instances, 
                                                                          const FmhaProblem& fmha_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<FmhaFwdSplitKVCodeGen> filtered_instances;
    
    // Score and rank instances based on multiple performance heuristics
    std::vector<std::pair<double, size_t>> scored_instances;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& tile_desc = instances[i].tile_desc_;
        double score = 0.0;
        
        // 1. Memory access efficiency (prioritize coalesced access patterns)
        int64_t total_block_size = tile_desc.m0_block_ * tile_desc.n0_block_ * tile_desc.k0_block_;
        int64_t warp_utilization = (tile_desc.m0_warp_ * tile_desc.n0_warp_ * tile_desc.k0_warp_);
        if (warp_utilization > 0) {
            score += std::log2(warp_utilization) * 0.3;  // Favor higher warp utilization
        }
        
        // 2. Register pressure estimation (avoid extreme values)
        int64_t reg_estimate = tile_desc.m0_warp_tile_ * tile_desc.n0_warp_tile_ * tile_desc.k0_warp_tile_;
        if (reg_estimate >= 32 && reg_estimate <= 512) {  // Sweet spot for register usage
            score += 0.25;
        }
        
        // 3. Problem size fitness
        int64_t seq_len = fmha_problem.max_seqlen_q_;
        int64_t head_dim = fmha_problem.hdim_q_;
        
        // Prefer tile sizes that divide evenly into problem dimensions
        if (seq_len % tile_desc.m0_block_ == 0) score += 0.2;
        if (head_dim % tile_desc.k0_block_ == 0) score += 0.2;
        
        // 4. Hardware efficiency (favor power-of-2 or multiple-of-32 sizes)
        auto is_efficient_size = [](int64_t size) {
            return (size % 32 == 0) || (size & (size - 1)) == 0;
        };
        
        if (is_efficient_size(tile_desc.m0_block_)) score += 0.1;
        if (is_efficient_size(tile_desc.n0_block_)) score += 0.1;
        if (is_efficient_size(tile_desc.k0_block_)) score += 0.1;
        
        scored_instances.emplace_back(score, i);
    }
    
    // Sort by score (highest first)
    std::sort(scored_instances.begin(), scored_instances.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top candidates (limit to reasonable number for heuristic mode)
    size_t max_candidates = std::min(static_cast<size_t>(16), instances.size());
    filtered_instances.reserve(max_candidates);
    
    for (size_t i = 0; i < max_candidates; ++i) {
        filtered_instances.push_back(instances[scored_instances[i].second]);
    }
    
    VLOG(2) << "FMHA split KV heuristic filter: reduced " << instances.size() 
            << " instances to " << filtered_instances.size() << " candidates";
    
    return filtered_instances;
}
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.m0_block_ > fmha_problem.q_seq_len_ || tile_desc.n0_block_ > fmha_problem.kv_seq_len_ ||
            tile_desc.n1_block_ > fmha_problem.v_head_dim_ || tile_desc.k0_max_block_ > fmha_problem.qk_head_dim_) {
            VLOG(3) << "Invalid FMHA tile descriptor: tile dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaFwdSplitKVEmitter::IsValidInstance(const FmhaFwdSplitKVCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
} 


// std::vector<FmhaFwdSplitKVTileDesc> FmhaFwdSplitKVEmitter::HeuristicFilter(const std::vector<FmhaFwdSplitKVTileDesc>& fmha_tile_desc,
//                                                        const FmhaProblem&               fmha_problem) const
// {
// }

// std::vector<FmhaFwdSplitKVAppendKVTileDesc> FmhaFwdSplitKVEmitter::HeuristicFilter(const std::vector<FmhaFwdSplitKVAppendKVTileDesc>& fmha_tile_desc,
//                                                                const FmhaProblem& fmha_problem) const
// {   
// }

// std::vector<FmhaFwdSplitKVSplitKVCombineTileDesc>
// FmhaFwdSplitKVEmitter::HeuristicFilter(const std::vector<FmhaFwdSplitKVSplitKVCombineTileDesc>& fmha_tile_desc,
//                              const FmhaProblem&                             fmha_problem) const
// {
// }


// Generate all possible FmhaFwdSplitKVCodeGen instances from a FmhaFwdSplitKVConfig
std::vector<FmhaFwdSplitKVCodeGen> FmhaFwdSplitKVEmitter::CreateInstanceForConfig(const FmhaFwdSplitKVConfig& config, const FmhaProblem& fmha_problem) {
    std::vector<FmhaFwdSplitKVCodeGen> result;

    std::vector<std::vector<int64_t>> all_lists = {
        // BlockConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.m0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.n0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.k0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.k0_max.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.n1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.k1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.m0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.n0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.k0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.m1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.n1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.k1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpTileConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.m0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.n0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.k0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.m1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.n1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.k1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.s.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.sk.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.d.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.dv.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.min_block_per_cu.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PipelineConfig (enum as int64_t)
        [&]{ std::vector<int64_t> v; for (const auto& x : config.pipeline.values) v.emplace_back(static_cast<int64_t>(GetBlockFmhaPipelineEnumFromString(x))); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t m0_block = vals[idx++];
        int64_t n0_block = vals[idx++];
        int64_t k0_block = vals[idx++];
        int64_t k0_max_block = vals[idx++];
        int64_t n1_block = vals[idx++];
        int64_t k1_block = vals[idx++];

        int64_t m0_warp = vals[idx++];
        int64_t n0_warp = vals[idx++];
        int64_t k0_warp = vals[idx++];
        int64_t m1_warp = vals[idx++];
        int64_t n1_warp = vals[idx++];
        int64_t k1_warp = vals[idx++];

        int64_t m0_warp_tile = vals[idx++];
        int64_t n0_warp_tile = vals[idx++];
        int64_t k0_warp_tile = vals[idx++];
        int64_t m1_warp_tile = vals[idx++];
        int64_t n1_warp_tile = vals[idx++];
        int64_t k1_warp_tile = vals[idx++];

        // PaddingConfig
        bool is_pad_q_seq_len_ = static_cast<bool>(vals[idx++]);
        bool is_pad_kv_seq_len_ = static_cast<bool>(vals[idx++]);
        bool is_pad_qk_head_dim_ = static_cast<bool>(vals[idx++]);
        bool is_pad_v_head_dim_ = static_cast<bool>(vals[idx++]);

        // launch config
        int64_t min_block_per_cu = vals[idx++];

        // PipelineConfig
        BlockFmhaPipelineEnum pipeline = static_cast<BlockFmhaPipelineEnum>(vals[idx++]);

        // Construct FmhaFwdSplitKVCodeGen
        FmhaFwdSplitKVCodeGen fmha;
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

void FmhaFwdSplitKVEmitter::GenerateInstances(FmhaProblem& fmha_problem)
{
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE == 0 || FLAGS_FC_TUNING_MODE == 1 || FLAGS_FC_TUNING_MODE == 2,
                  true,
                  Unavailable("Unsupported mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", FLAGS_FC_TUNING_MODE));


    // Check if instances already exist for this FMHA kind
    if (instance_map_.find(fmha_problem.kind_) != instance_map_.end() && !instance_map_[fmha_problem.kind_].empty()) {
        VLOG(2) << "Instances already generated for FMHA kind: " << GetFmhaKindName(fmha_problem.kind_);
        return;
    }

    // Load legacy GEMM configuration if available
    std::vector<FmhaFwdSplitKVCodeGen> fmha_instances;
    if (FLAGS_FC_ENABLE_CONFIG_JSON) {
        auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetFmhaKindName(fmha_problem.kind_);
        if(FLAGS_FC_ENABLE_JSON_MODE == 0) {
            std::filesystem::path json_path = base_json_path / "default_config.json";
            FmhaFwdSplitKVConfig config = LoadDefaultConfigJson<FmhaFwdSplitKVConfig>(json_path);
            fmha_instances = CreateInstanceForConfig(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 1) {
            std::filesystem::path json_path = base_json_path / "user_config.json";
            FmhaFwdSplitKVConfig config = LoadDefaultConfigJson<FmhaFwdSplitKVConfig>(json_path);
            fmha_instances = CreateInstanceForConfig(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 2) {
            std::filesystem::path default_json_path = base_json_path / "default_config.json";
            FmhaFwdSplitKVConfig default_config = LoadDefaultConfigJson<FmhaFwdSplitKVConfig>(default_json_path);
            auto gemm_default_instances = CreateInstanceForConfig(default_config, fmha_problem);

            std::filesystem::path user_json_path = base_json_path / "user_config.json";
            FmhaFwdSplitKVConfig user_config = LoadDefaultConfigJson<FmhaFwdSplitKVConfig>(user_json_path);
            auto gemm_user_instances = CreateInstanceForConfig(user_config, fmha_problem);

            fmha_instances.insert(fmha_instances.end(), gemm_default_instances.begin(), gemm_default_instances.end());
            fmha_instances.insert(fmha_instances.end(), gemm_user_instances.begin(), gemm_user_instances.end());
        }
    else{
            LOG(WARNING)<< "FC_ENABLE_JSON_MODE is set to an unsupported value: " << FLAGS_FC_ENABLE_JSON_MODE;
        }
    } else {
        LOG(WARNING)<< "FC_ENABLE_CONFIG_JSON is not enabled";
    }

    for (const auto& config : g_backup_fmha_fwd_split_kv_config) {
        FmhaFwdSplitKVCodeGen fmha;

        fmha.problem_ = fmha_problem;

        fmha.tile_desc_ = FmhaFwdSplitKVTileDesc{
            config.tile_shape.block_tile.m0.values[0],
            config.tile_shape.block_tile.n0.values[0],
            config.tile_shape.block_tile.k0.values[0],
            config.tile_shape.block_tile.k0_max.values[0],
            config.tile_shape.block_tile.n1.values[0],
            config.tile_shape.block_tile.k1.values[0],
            config.tile_shape.block_warps.m0.values[0],
            config.tile_shape.block_warps.n0.values[0],
            config.tile_shape.block_warps.k0.values[0],
            config.tile_shape.block_warps.m1.values[0],
            config.tile_shape.block_warps.n1.values[0],
            config.tile_shape.block_warps.k1.values[0],
            config.tile_shape.warp_tile.m0.values[0],
            config.tile_shape.warp_tile.n0.values[0],
            config.tile_shape.warp_tile.k0.values[0],
            config.tile_shape.warp_tile.m1.values[0],
            config.tile_shape.warp_tile.n1.values[0],
            config.tile_shape.warp_tile.k1.values[0]
        };

        fmha.is_pad_q_seq_len_ = config.padding.s.values[0];
        fmha.is_pad_kv_seq_len_ = config.padding.sk.values[0];
        fmha.is_pad_qk_head_dim_ = config.padding.d.values[0];
        fmha.is_pad_v_head_dim_ = config.padding.dv.values[0];

        fmha.min_block_per_cu_ = config.launch.min_block_per_cu.values[0];

        fmha.pipeline_ = GetBlockFmhaPipelineEnumFromString(config.pipeline.values[0]);

        fmha_instances.push_back(fmha);
    }

    // check instances
    std::vector<FmhaFwdSplitKVCodeGen> valid_fmha_instances;
    for (const auto& fmha_instance : fmha_instances) {
        if (IsValidInstance(fmha_instance)) {
            valid_fmha_instances.push_back(fmha_instance);
        }
    }

    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {
            // Heuristic mode
        }
        case 1: {
            VLOG(1) << "Generating instances using autotuning mode for FMHA kind: "
                    << GetFmhaKindName(fmha_problem.kind_);
            break;
        }
        case 2: {
            VLOG(1) << "Generating instances using hybrid mode for FMHA kind: " << GetFmhaKindName(fmha_problem.kind_);
            break;
        }
        default:
            FC_THROW(Unavailable("Invalid mode:{} ", FLAGS_FC_TUNING_MODE));
    }


    if (valid_fmha_instances.empty()) {
        FC_THROW(Unavailable("No valid FMHA instances found for FMHA problem"));
    }

    // Generate instances
    auto& kind_instance_map = instance_map_[fmha_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& instance : valid_fmha_instances) {
        try {
            std::string instance_name = instance.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = std::move(instance);
                generated_count++;
                VLOG(2) << "Generated FMHA instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate FMHA instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA codegen for instance: " << instance.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " FMHA instances for kind: " << GetFmhaKindName(fmha_problem.kind_)
            << " (total: " << num_instances_ << ")";
}

void FmhaFwdSplitKVEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
