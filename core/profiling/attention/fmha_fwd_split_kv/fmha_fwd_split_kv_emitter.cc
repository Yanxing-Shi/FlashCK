#include "core/profiling/attention/fmha_fwd_split_kv/fmha_fwd_split_kv_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files

namespace flashck {

bool FmhaFwdSplitKVEmitter::IsValidTile(const FmhaFwdSplitKVTileDesc& tile_desc, const FmhaFwdSplitKVProblem& fmha_fwd_split_kv_problem)
{
    // Validate all tile parameters are positive for split-KV operations
    if (tile_desc.m0_block_ <= 0 || tile_desc.n0_block_ <= 0 || tile_desc.k0_block_ <= 0 || 
        tile_desc.k0_max_block_ <= 0 || tile_desc.n1_block_ <= 0 || tile_desc.k1_block_ <= 0 ||
        tile_desc.m0_warp_ <= 0 || tile_desc.n0_warp_ <= 0 || tile_desc.k0_warp_ < 0 ||
        tile_desc.m1_warp_ <= 0 || tile_desc.n1_warp_ <= 0 || tile_desc.k1_warp_ < 0 ||
        tile_desc.m0_warp_tile_ <= 0 || tile_desc.n0_warp_tile_ <= 0 || tile_desc.k0_warp_tile_ <= 0 ||
        tile_desc.m1_warp_tile_ <= 0 || tile_desc.n1_warp_tile_ <= 0 || tile_desc.k1_warp_tile_ <= 0) {
        VLOG(3) << "Invalid FMHA split-KV tile: negative or zero values not allowed";
        return false;
    }

    // Validate k0_block_ <= k0_max_block_ for split-KV constraints
    if (tile_desc.k0_block_ > tile_desc.k0_max_block_) {
        VLOG(3) << "Invalid FMHA split-KV tile: k0_block_ > k0_max_block_";
        return false;
    }

    // Validate warp*warp_tile <= block sizes for split-KV operations
    if (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n0_warp_ * tile_desc.n0_warp_tile_ > tile_desc.n0_block_ ||
        tile_desc.k0_warp_ * tile_desc.k0_warp_tile_ > tile_desc.k0_block_ ||
        tile_desc.m1_warp_ * tile_desc.m1_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n1_warp_ * tile_desc.n1_warp_tile_ > tile_desc.n1_block_ ||
        tile_desc.k1_warp_ * tile_desc.k1_warp_tile_ > tile_desc.k1_block_) {
        VLOG(3) << "Invalid FMHA split-KV tile: warp*warp_tile exceeds block size";
        return false;
    }

    // Validate block sizes are divisible by warp*warp_tile for split-KV
    if ((tile_desc.m0_block_ % (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_) != 0) ||
        (tile_desc.n0_block_ % (tile_desc.n0_warp_ * tile_desc.n0_warp_tile_) != 0) ||
        (tile_desc.k0_block_ % (std::max<int64_t>(1, tile_desc.k0_warp_ * tile_desc.k0_warp_tile_)) != 0) ||
        (tile_desc.m0_block_ % (tile_desc.m1_warp_ * tile_desc.m1_warp_tile_) != 0) ||
        (tile_desc.n1_block_ % (tile_desc.n1_warp_ * tile_desc.n1_warp_tile_) != 0) ||
        (tile_desc.k1_block_ % (std::max<int64_t>(1, tile_desc.k1_warp_ * tile_desc.k1_warp_tile_)) != 0)) {
        VLOG(3) << "Invalid FMHA split-KV tile: block size not divisible by warp*warp_tile";
        return false;
    }

    // Validate against problem dimensions for Batch mode with split-KV considerations
    if (fmha_fwd_split_kv_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.m0_block_ > fmha_fwd_split_kv_problem.q_seq_len_ || 
            tile_desc.n0_block_ > fmha_fwd_split_kv_problem.kv_seq_len_ ||
            tile_desc.n1_block_ > fmha_fwd_split_kv_problem.v_head_dim_ || 
            tile_desc.k0_max_block_ > fmha_fwd_split_kv_problem.qk_head_dim_) {
            VLOG(3) << "Invalid FMHA split-KV tile: dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaFwdSplitKVEmitter::IsValidInstance(const FmhaFwdSplitKVCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
}

std::vector<FmhaFwdSplitKVCodeGen> FmhaFwdSplitKVEmitter::HeuristicFilter(
    const std::vector<FmhaFwdSplitKVCodeGen>& instances,
    const FmhaFwdSplitKVProblem& fmha_fwd_split_kv_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<FmhaFwdSplitKVCodeGen> filtered;
    
    // Split-KV specific heuristics
    // Heuristic 1: Prefer configurations that optimize K-V splitting efficiency
    constexpr int64_t preferred_split_factor = 4;
    
    // Heuristic 2: Balance memory bandwidth for split operations
    constexpr int64_t preferred_kv_block_size = 256;
    
    // Heuristic 3: Optimize for split-KV memory access patterns
    const int64_t kv_seq_len = fmha_fwd_split_kv_problem.kv_seq_len_;
    const int64_t optimal_split_size = kv_seq_len / preferred_split_factor;
    
    for (const auto& instance : instances) {
        const auto& tile = instance.tile_desc_;
        
        // Filter 1: Skip configurations with poor split-KV efficiency
        if (tile.n0_block_ < 32 || tile.k0_block_ < 16) {
            continue;
        }
        
        // Filter 2: Prefer configurations optimized for K-V splitting
        int64_t kv_efficiency_score = 0;
        if (tile.n0_block_ <= optimal_split_size * 2 && tile.n0_block_ >= optimal_split_size / 2) {
            kv_efficiency_score += 2; // Good split granularity
        }
        if (tile.k0_block_ % 32 == 0) {
            kv_efficiency_score += 1; // Memory coalescing friendly
        }
        
        // Filter 3: Ensure reasonable warp utilization for split-KV
        int64_t total_warps = tile.m0_warp_ * tile.n0_warp_ * tile.k0_warp_;
        if (total_warps > 16 || total_warps < 1) {
            continue;
        }
        
        // Filter 4: Split-KV specific memory access optimization
        if (tile.k0_block_ * tile.n0_block_ < 1024) {
            continue; // Too small for efficient split operations
        }
        
        // Filter 5: Avoid excessive head dimension splitting in split-KV
        if (tile.k0_block_ < fmha_fwd_split_kv_problem.qk_head_dim_ / 8) {
            continue;
        }
        
        // Only keep instances that pass split-KV specific criteria
        if (kv_efficiency_score >= 1) {
            filtered.push_back(instance);
        }
    }
    
    // If filtering is too aggressive, return a subset with relaxed criteria
    if (filtered.empty()) {
        VLOG(2) << "Split-KV heuristic filter too aggressive, returning subset of original instances";
        const size_t subset_size = std::min<size_t>(instances.size(), 8);
        filtered.assign(instances.begin(), instances.begin() + subset_size);
    }
    
    VLOG(2) << "Split-KV heuristic filter: " << instances.size() << " -> " << filtered.size() << " instances";
    return filtered;
}

std::vector<FmhaFwdSplitKVCodeGen> FmhaFwdSplitKVEmitter::CreateInstanceForConfig(
    const FmhaFwdSplitKVConfig& config, const FmhaFwdSplitKVProblem& fmha_fwd_split_kv_problem) 
{
    std::vector<FmhaFwdSplitKVCodeGen> result;

    // Convert all config parameters to int64_t vectors for CartesianProduct
    std::vector<std::vector<int64_t>> all_param_lists = {
        // Block tile configuration (6 parameters) - using GetAllValues() for both modes compatibility
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.m0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.n0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.k0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.k0_max.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.n1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.k1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Block warp configuration (6 parameters)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.m0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.n0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.k0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.m1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.n1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.k1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Warp tile configuration (6 parameters)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.m0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.n0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.k0.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.m1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.n1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.k1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Padding configuration (4 parameters, bool->int64_t)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.padding.s.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.padding.sk.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.padding.d.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.padding.dv.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // has_uneven_splits (bool->int64_t)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.has_uneven_splits.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // merge_groups_num_head_q_seq_len (bool->int64_t)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.merge_groups_num_head_q_seq_len.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Pipeline configuration (string->enum->int64_t)
        [&]{ std::vector<int64_t> v; 
             for (const auto& x : config.strategy.pipeline.GetAllValues()) 
                 v.push_back(static_cast<int64_t>(GetBlockFmhaPipelineEnumFromString(x))); 
             return v; }(),
        
        // num splits
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.strategy.num_splits.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Launch configuration (2 parameter)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.max_thread_per_block.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.min_block_per_cu.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
    };

    // Generate all parameter combinations using CartesianProduct
    CartesianProduct(all_param_lists, [&](const std::vector<int64_t>& param_values) {
        size_t idx = 0;
        
        // Extract block tile parameters
        int64_t m0_block = param_values[idx++];
        int64_t n0_block = param_values[idx++];
        int64_t k0_block = param_values[idx++];
        int64_t k0_max_block = param_values[idx++];
        int64_t n1_block = param_values[idx++];
        int64_t k1_block = param_values[idx++];
        
        // Extract block warp parameters
        int64_t m0_warp = param_values[idx++];
        int64_t n0_warp = param_values[idx++];
        int64_t k0_warp = param_values[idx++];
        int64_t m1_warp = param_values[idx++];
        int64_t n1_warp = param_values[idx++];
        int64_t k1_warp = param_values[idx++];
        
        // Extract warp tile parameters
        int64_t m0_warp_tile = param_values[idx++];
        int64_t n0_warp_tile = param_values[idx++];
        int64_t k0_warp_tile = param_values[idx++];
        int64_t m1_warp_tile = param_values[idx++];
        int64_t n1_warp_tile = param_values[idx++];
        int64_t k1_warp_tile = param_values[idx++];
        
        // Extract padding parameters
        bool is_pad_q_seq_len = static_cast<bool>(param_values[idx++]);
        bool is_pad_kv_seq_len = static_cast<bool>(param_values[idx++]);
        bool is_pad_qk_head_dim = static_cast<bool>(param_values[idx++]);
        bool is_pad_v_head_dim = static_cast<bool>(param_values[idx++]);

        // Extract has_uneven_splits
        bool has_uneven_splits = static_cast<bool>(param_values[idx++]);

        // Extract merge_groups_num_head_q_seq_len
        bool merge_groups_num_head_q_seq_len = static_cast<bool>(param_values[idx++]);
        
        // Extract pipeline parameters
        BlockFmhaPipelineEnum pipeline = static_cast<BlockFmhaPipelineEnum>(param_values[idx++]);

        // Extract num_splits
        int64_t num_splits = param_values[idx++];

        // Extract launch parameters
        int64_t max_thread_per_block = param_values[idx++];
        int64_t min_block_per_cu = param_values[idx++];

        // Construct FmhaFwdSplitKVCodeGen instance
        FmhaFwdSplitKVCodeGen instance;
        instance.problem_ = fmha_fwd_split_kv_problem;
        
        // Set tile descriptor
        instance.tile_desc_.m0_block_ = m0_block;
        instance.tile_desc_.n0_block_ = n0_block;
        instance.tile_desc_.k0_block_ = k0_block;
        instance.tile_desc_.k0_max_block_ = k0_max_block;
        instance.tile_desc_.n1_block_ = n1_block;
        instance.tile_desc_.k1_block_ = k1_block;
        instance.tile_desc_.m0_warp_ = m0_warp;
        instance.tile_desc_.n0_warp_ = n0_warp;
        instance.tile_desc_.k0_warp_ = k0_warp;
        instance.tile_desc_.m1_warp_ = m1_warp;
        instance.tile_desc_.n1_warp_ = n1_warp;
        instance.tile_desc_.k1_warp_ = k1_warp;
        instance.tile_desc_.m0_warp_tile_ = m0_warp_tile;
        instance.tile_desc_.n0_warp_tile_ = n0_warp_tile;
        instance.tile_desc_.k0_warp_tile_ = k0_warp_tile;
        instance.tile_desc_.m1_warp_tile_ = m1_warp_tile;
        instance.tile_desc_.n1_warp_tile_ = n1_warp_tile;
        instance.tile_desc_.k1_warp_tile_ = k1_warp_tile;
        
        // Set padding configuration
        instance.is_pad_q_seq_len_ = is_pad_q_seq_len;
        instance.is_pad_kv_seq_len_ = is_pad_kv_seq_len;
        instance.is_pad_qk_head_dim_ = is_pad_qk_head_dim;
        instance.is_pad_v_head_dim_ = is_pad_v_head_dim;
        
        // Set has_uneven_splits
        instance.has_uneven_splits_ = has_uneven_splits;

        // Set merge_groups_num_head_q_seq_len
        instance.merge_groups_num_head_q_seq_len_ = merge_groups_num_head_q_seq_len;

        // Set pipeline configuration
        instance.pipeline_ = pipeline;
        
        // Set num_splits
        instance.num_splits_ = num_splits;

        // Set launch configuration
        instance.min_block_per_cu_ = min_block_per_cu;
        instance.max_thread_per_block_ = max_thread_per_block;
        
        result.push_back(instance);
    });

    return result;
}

void FmhaFwdSplitKVEmitter::GenerateInstances(FmhaFwdSplitKVProblem& fmha_fwd_split_kv_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid tuning mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", 
                              FLAGS_FC_TUNING_MODE));

    VLOG(1) << "Generating FMHA split-KV instances for mode: " << FLAGS_FC_TUNING_MODE;

    // Load configurations from JSON files
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / "attention" / "fmha_fwd_split_kv";
    std::vector<FmhaFwdSplitKVCodeGen> all_instances;
    
    // Load backup configurations (pre-validated, single-value configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        std::filesystem::path json_path = base_json_path / "backup_config.json";
        try {
            auto backup_configs = LoadConfigJson<std::vector<FmhaFwdSplitKVConfig>>(json_path);
            for (const auto& config : backup_configs) {
                auto instances = CreateInstanceForConfig(config, fmha_fwd_split_kv_problem);
                all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            }
            VLOG(2) << "Loaded " << backup_configs.size() << " split-KV backup configurations";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load split-KV backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges for tuning)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        std::filesystem::path json_path = base_json_path / "default_config.json";
        try {
            auto default_config = LoadConfigJson<FmhaFwdSplitKVConfig>(json_path);
            auto instances = CreateInstanceForConfig(default_config, fmha_fwd_split_kv_problem);
            all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            VLOG(2) << "Loaded split-KV default configuration with " << instances.size() << " instances";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load split-KV default config: " << e.what();
        }
    }

    // Load user configurations (custom user-defined configs)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        std::filesystem::path json_path = base_json_path / "user_config.json";
        try {
            auto user_config = LoadConfigJson<FmhaFwdSplitKVConfig>(json_path);
            auto instances = CreateInstanceForConfig(user_config, fmha_fwd_split_kv_problem);
            all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            VLOG(2) << "Loaded split-KV user configuration with " << instances.size() << " instances";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load split-KV user config: " << e.what();
        }
    }

    // Filter out invalid instances
    std::vector<FmhaFwdSplitKVCodeGen> valid_instances;
    for (const auto& instance : all_instances) {
        if (IsValidInstance(instance)) {
            valid_instances.push_back(instance);
        }
    }

    VLOG(2) << "Split-KV validation: " << all_instances.size() << " -> " << valid_instances.size() << " valid instances";

    if (valid_instances.empty()) {
        FC_THROW(Unavailable("No valid FMHA split-KV instances found"));
    }

    // Apply mode-specific strategy
    std::vector<FmhaFwdSplitKVCodeGen> final_instances;
    std::random_device rd;
    std::mt19937 rng(rd());
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {
            // Heuristic mode: filter + random selection
            final_instances = HeuristicFilter(valid_instances, fmha_fwd_split_kv_problem);
            if (!final_instances.empty()) {
                // Randomly select one instance for fast execution
                std::uniform_int_distribution<> dist(0, final_instances.size() - 1);
                auto selected = final_instances[dist(rng)];
                final_instances = {selected};
                VLOG(1) << "Split-KV heuristic mode: selected 1 instance from " << valid_instances.size();
            }
            break;
        }
        case 1: {
            // Autotuning mode: use all valid instances
            final_instances = valid_instances;
            VLOG(1) << "Split-KV autotuning mode: using all " << final_instances.size() << " valid instances";
            break;
        }
        case 2: {
            // Hybrid mode: combine heuristic filtering + all instances
            auto heuristic_instances = HeuristicFilter(valid_instances, fmha_fwd_split_kv_problem);
            final_instances = heuristic_instances;
            final_instances.insert(final_instances.end(), valid_instances.begin(), valid_instances.end());
            
            // Remove duplicates
            std::sort(final_instances.begin(), final_instances.end(), 
                     []( auto& a,  auto& b) {
                         return a.GetInstanceName() < b.GetInstanceName();
                     });
            final_instances.erase(std::unique(final_instances.begin(), final_instances.end(),
                                            []( auto& a,  auto& b) {
                                                return a.GetInstanceName() == b.GetInstanceName();
                                            }), final_instances.end());
            
            VLOG(1) << "Split-KV hybrid mode: using " << final_instances.size() << " unique instances";
            break;
        }
        default:
            FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
    }

    if (final_instances.empty()) {
        FC_THROW(Unavailable("No final FMHA split-KV instances after mode-specific filtering"));
    }

    // Store instances in the map
    int64_t generated_count = 0;

    for (auto& instance : final_instances) {
        try {
            std::string instance_name = instance.GetInstanceName();
            
            // Avoid duplicates
            if (instance_map_.find(instance_name) == instance_map_.end()) {
                instance_map_[instance_name] = instance;
                generated_count++;
                VLOG(3) << "Generated FMHA split-KV instance: " << instance_name;
            } else {
                VLOG(3) << "Skipped duplicate FMHA split-KV instance: " << instance_name;
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA split-KV codegen for instance: " << instance.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " FMHA split-KV instances " 
            << " (total: " << num_instances_ << ")";
}

void FmhaFwdSplitKVEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
    VLOG(2) << "Cleared all FMHA split-KV instances";
}

}  // namespace flashck
