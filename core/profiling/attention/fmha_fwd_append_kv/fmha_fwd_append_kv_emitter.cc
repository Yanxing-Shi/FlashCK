#include "core/profiling/attention/fmha_fwd_append_kv/fmha_fwd_append_kv_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);     // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);    // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);   // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);      // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);    // Base path for config files

namespace flashck {

bool FmhaFwdAppendKVEmitter::IsValidTile(const FmhaFwdAppendKVTileDesc& tile_desc, const FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem)
{
    // Validate all tile parameters are positive
    if (tile_desc.s_block_ <= 0 || tile_desc.sk_block_ <= 0 || 
        tile_desc.d_block_ <= 0 || tile_desc.dv_block_ <= 0) {
        VLOG(3) << "Invalid FMHA fwd append KV tile: negative or zero values not allowed";
        return false;
    }

    return true;
}

bool FmhaFwdAppendKVEmitter::IsValidInstance(const FmhaFwdAppendKVCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
} 

std::vector<FmhaFwdAppendKVCodeGen> FmhaFwdAppendKVEmitter::HeuristicFilter(const std::vector<FmhaFwdAppendKVCodeGen>& instances, 
                                                                           const FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<FmhaFwdAppendKVCodeGen> filtered_instances;
    
    // Score and rank instances based on multiple performance heuristics
    std::vector<std::pair<double, size_t>> scored_instances;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& tile_desc = instances[i].tile_desc_;
        double score = 0.0;
        
        // 1. Memory access efficiency (prioritize balanced block sizes)
        int64_t total_block_size = tile_desc.s_block_ * tile_desc.sk_block_ * tile_desc.d_block_;
        score += std::log2(std::max<int64_t>(1, total_block_size)) * 0.3;
        
        // 2. Problem size fitness
        int64_t seq_len = fmha_fwd_append_kv_problem.q_seq_len_;
        int64_t head_dim = fmha_fwd_append_kv_problem.qk_head_dim_;
        
        // Prefer tile sizes that divide evenly into problem dimensions
        if (seq_len % tile_desc.s_block_ == 0) score += 0.25;
        if (head_dim % tile_desc.d_block_ == 0) score += 0.25;
        
        // 3. Hardware efficiency (favor power-of-2 or multiple-of-32 sizes)
        auto is_efficient_size = [](int64_t size) {
            return (size % 32 == 0) || (size & (size - 1)) == 0;
        };
        
        if (is_efficient_size(tile_desc.s_block_)) score += 0.1;
        if (is_efficient_size(tile_desc.sk_block_)) score += 0.1;
        if (is_efficient_size(tile_desc.d_block_)) score += 0.1;
        
        scored_instances.emplace_back(score, i);
    }
    
    // Sort by score (highest first)
    std::sort(scored_instances.begin(), scored_instances.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top candidates (limit to reasonable number for heuristic mode)
    size_t max_candidates = std::min(static_cast<size_t>(12), instances.size());
    filtered_instances.reserve(max_candidates);
    
    for (size_t i = 0; i < max_candidates; ++i) {
        filtered_instances.push_back(instances[scored_instances[i].second]);
    }
    
    VLOG(2) << "FMHA append KV heuristic filter: reduced " << instances.size() 
            << " instances to " << filtered_instances.size() << " candidates";
    
    return filtered_instances;
}

// Generate all possible FmhaFwdAppendKVCodeGen instances from a FmhaFwdAppendKVConfig
std::vector<FmhaFwdAppendKVCodeGen> FmhaFwdAppendKVEmitter::CreateInstanceForConfig(const FmhaFwdAppendKVConfig& config, const FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem) {
    std::vector<FmhaFwdAppendKVCodeGen> result;

    std::vector<std::vector<int64_t>> all_lists = {
        // BlockConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.s.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.sk.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.d.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.dv.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<int64_t> v; for (auto x : config.trait.padding.s.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.trait.padding.sk.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.trait.padding.d.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.trait.padding.dv.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.max_thread_per_block.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.min_block_per_cu.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t s_block = vals[idx++];
        int64_t sk_block = vals[idx++];
        int64_t d_block = vals[idx++];
        int64_t dv_block = vals[idx++];

        // PaddingConfig
        bool is_pad_q_seq_len = static_cast<bool>(vals[idx++]);
        bool is_pad_kv_seq_len = static_cast<bool>(vals[idx++]);
        bool is_pad_qk_head_dim = static_cast<bool>(vals[idx++]);
        bool is_pad_v_head_dim = static_cast<bool>(vals[idx++]);

        // launch config
        int64_t max_thread_per_block = vals[idx++];
        int64_t min_block_per_cu = vals[idx++];

        // Construct FmhaFwdAppendKVCodeGen
        FmhaFwdAppendKVCodeGen fmha;
        fmha.problem_ = fmha_fwd_append_kv_problem;
        // tile_desc
        fmha.tile_desc_.s_block_ = s_block;
        fmha.tile_desc_.sk_block_ = sk_block;
        fmha.tile_desc_.d_block_ = d_block;
        fmha.tile_desc_.dv_block_ = dv_block;

        // Padding
        fmha.is_pad_q_seq_len_ = is_pad_q_seq_len;
        fmha.is_pad_kv_seq_len_ = is_pad_kv_seq_len;
        fmha.is_pad_qk_head_dim_ = is_pad_qk_head_dim;
        fmha.is_pad_v_head_dim_ = is_pad_v_head_dim;

        // Launch
        fmha.max_thread_per_block_ = max_thread_per_block;
        fmha.min_block_per_cu_ = min_block_per_cu;
        result.push_back(fmha);
    });

    return result;
}

void FmhaFwdAppendKVEmitter::GenerateInstances(FmhaFwdAppendKVProblem& fmha_fwd_append_kv_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid FC_TUNING_MODE: {}. Valid values: 0(heuristic), 1(autotuning), 2(hybrid)", 
                             FLAGS_FC_TUNING_MODE));

    std::vector<FmhaFwdAppendKVCodeGen> all_instances;

    // Configuration loading based on enabled flags
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / "attention" / "fmha_fwd_append_kv";

    // Load backup configurations (pre-validated single configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        try {
            std::filesystem::path backup_path = base_json_path / "backup_config.json";
            if (std::filesystem::exists(backup_path)) {
                auto backup_configs = LoadConfigJson<std::vector<FmhaFwdAppendKVConfig>>(backup_path);
                for (const auto& config : backup_configs) {
                    auto backup_instances = CreateInstanceForConfig(config, fmha_fwd_append_kv_problem);
                    all_instances.insert(all_instances.end(), backup_instances.begin(), backup_instances.end());
                }
                VLOG(2) << "Loaded " << backup_configs.size() << " FMHA append KV backup configurations";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load FMHA append KV backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        try {
            std::filesystem::path default_path = base_json_path / "default_config.json";
            if (std::filesystem::exists(default_path)) {
                auto default_config = LoadConfigJson<FmhaFwdAppendKVConfig>(default_path);
                auto default_instances = CreateInstanceForConfig(default_config, fmha_fwd_append_kv_problem);
                all_instances.insert(all_instances.end(), default_instances.begin(), default_instances.end());
                VLOG(2) << "Loaded FMHA append KV default configuration with " << default_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load FMHA append KV default config: " << e.what();
        }
    }

    // Load user configurations (custom parameter ranges)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        try {
            std::filesystem::path user_path = base_json_path / "user_config.json";
            if (std::filesystem::exists(user_path)) {
                auto user_config = LoadConfigJson<FmhaFwdAppendKVConfig>(user_path);
                auto user_instances = CreateInstanceForConfig(user_config, fmha_fwd_append_kv_problem);
                all_instances.insert(all_instances.end(), user_instances.begin(), user_instances.end());
                VLOG(2) << "Loaded FMHA append KV user configuration with " << user_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load FMHA append KV user config: " << e.what();
        }
    }

    // Apply mode-specific processing
    std::vector<FmhaFwdAppendKVCodeGen> final_instances;
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {  // Heuristic mode: filter + random selection for fast execution
            auto filtered_instances = HeuristicFilter(all_instances, fmha_fwd_append_kv_problem);
            if (!filtered_instances.empty()) {
                // Randomly select one optimal configuration for fast execution
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, filtered_instances.size() - 1);
                final_instances.push_back(filtered_instances[dis(gen)]);
                VLOG(1) << "FMHA append KV heuristic mode: selected 1 instance from " 
                        << filtered_instances.size() << " filtered candidates";
            }
            break;
        }
        case 1: {  // Autotuning mode: use all valid instances for comprehensive search
            final_instances = all_instances;
            VLOG(1) << "FMHA append KV autotuning mode: using all " << all_instances.size() << " instances";
            break;
        }
        case 2: {  // Hybrid mode: heuristic filtering + broader search
            auto filtered_instances = HeuristicFilter(all_instances, fmha_fwd_append_kv_problem);
            final_instances = filtered_instances.empty() ? all_instances : filtered_instances;
            VLOG(1) << "FMHA append KV hybrid mode: using " << final_instances.size() 
                    << " instances (filtered from " << all_instances.size() << ")";
            break;
        }
    }

    // Validate and store instances
    num_instances_ = 0;
    for (auto& instance : final_instances) {
        if (IsValidInstance(instance)) {
            instance_map_[instance.GetInstanceName()] = instance;
            ++num_instances_;
        }
    }

    VLOG(1) << "Generated " << num_instances_ << " valid FMHA append KV instances " 
            << " (mode " << FLAGS_FC_TUNING_MODE << ")";
}

void FmhaFwdAppendKVEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
