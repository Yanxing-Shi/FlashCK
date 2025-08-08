#include "core/profiling/moe/moe_sorting_ex/moe_sorting_ex_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files

namespace flashck {

bool MoeSortingExEmitter::IsValidInstance(const MoeSortingExCodeGen& instance)
{
    return true;
}

std::vector<MoeSortingExCodeGen> MoeSortingExEmitter::HeuristicFilter(
    const std::vector<MoeSortingExCodeGen>& instances,
    const MoeSortingExProblem& moe_sorting_ex_problem)
{
    if (instances.empty()) {
        return {};
    }

    // std::vector<MoeSortingExCodeGen> filtered;
    
    // // Score and rank instances based on sorting performance heuristics
    // std::vector<std::pair<double, size_t>> scored_instances;
    
    // for (size_t i = 0; i < instances.size(); ++i) {
    //     const auto& instance = instances[i];
    //     double score = 0.0;
        
    //     // 1. Memory bandwidth efficiency (prefer moderate unroll factors)
    //     if (instance.load_unroll_ >= 2 && instance.load_unroll_ <= 8) {
    //         score += 0.3;  // Good memory coalescing
    //     }
        
    //     // 2. Expert tile size efficiency
    //     const double expert_utilization = static_cast<double>(instance.expert_tile_) / moe_sorting_ex_problem.num_experts_;
    //     if (expert_utilization >= 0.25 && expert_utilization <= 1.0) {
    //         score += 0.25;  // Good expert coverage
    //     }
        
    //     // 3. Load balancing considerations
    //     const int64_t tokens_per_expert = moe_sorting_ex_problem.input_tokens_ * moe_sorting_ex_problem.topk_ / moe_sorting_ex_problem.num_experts_;
    //     const int64_t work_per_tile = tokens_per_expert * instance.expert_tile_;
    //     if (work_per_tile >= 256 && work_per_tile <= 4096) {
    //         score += 0.2;  // Good workload granularity
    //     }
        
    //     // 4. Block occupancy optimization
    //     if (instance.min_block_per_cu_ >= 1 && instance.min_block_per_cu_ <= 4) {
    //         score += 0.15;  // Good occupancy for sorting
    //     }
        
    //     // 5. Problem size adaptation
    //     if (moe_sorting_ex_problem.input_tokens_ >= 1024) {
    //         // For large problems, prefer higher unroll
    //         if (instance.load_unroll_ >= 4) score += 0.1;
    //     } else {
    //         // For small problems, prefer lower unroll
    //         if (instance.load_unroll_ <= 4) score += 0.1;
    //     }
        
    //     scored_instances.emplace_back(score, i);
    // }
    
    // // Sort by score (highest first)
    // std::sort(scored_instances.begin(), scored_instances.end(), 
    //           [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // // Select top candidates (limit to reasonable number for heuristic mode)
    // size_t max_candidates = std::min(static_cast<size_t>(8), instances.size());
    // filtered.reserve(max_candidates);
    
    // for (size_t i = 0; i < max_candidates; ++i) {
    //     filtered.push_back(instances[scored_instances[i].second]);
    // }
    
    // VLOG(2) << "MoE sorting heuristic filter: reduced " << instances.size() 
    //         << " instances to " << filtered.size() << " candidates";
    
    // return filtered;
    return {};
}

std::vector<MoeSortingExCodeGen> MoeSortingExEmitter::CreateInstanceForConfig(
    const MoeSortingExConfig& config, const MoeSortingExProblem& moe_sorting_ex_problem) 
{
    std::vector<MoeSortingExCodeGen> result;

    // Convert all config parameters to int64_t vectors for CartesianProduct
    std::vector<std::vector<int64_t>> all_param_lists = {
        // Trait configuration
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.sub_token_tile.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.sub_token_one_shot.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.local_expert_masking.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.local_token.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.skip_expert_with_zero_token.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.expert_tile.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        // Launch configuration
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.max_thread_per_block.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.min_block_per_cu.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
    };

    CartesianProduct(all_param_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        
        // Sorting configuration
        int64_t sub_token_tile = vals[idx++];
        bool sub_token_one_shot = static_cast<bool>(vals[idx++]);
        bool local_token_expert_masking = static_cast<bool>(vals[idx++]);
        bool local_token = static_cast<bool>(vals[idx++]);
        bool skip_expert_with_zero_token = static_cast<bool>(vals[idx++]);
        int64_t expert_tile = vals[idx++];
        
        // Launch configuration
        int64_t max_thread_per_block = vals[idx++];
        int64_t min_block_per_cu = vals[idx++];

        // Construct MoeSortingExCodeGen
        MoeSortingExCodeGen instance;
        instance.problem_ = moe_sorting_ex_problem;
        instance.sub_token_tile_ = sub_token_tile;
        instance.sub_token_one_shot_ = sub_token_one_shot;
        instance.local_token_expert_masking_ = local_token_expert_masking;
        instance.local_token_ = local_token;
        instance.skip_expert_with_zero_token_ = skip_expert_with_zero_token;
        instance.expert_tile_ = expert_tile;
        instance.max_thread_per_block_ = max_thread_per_block;
        instance.min_block_per_cu_ = min_block_per_cu;
        
        result.push_back(instance);
    });
    
    return result;
}

void MoeSortingExEmitter::GenerateInstances(MoeSortingExProblem& moe_sorting_ex_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid FC_TUNING_MODE: {}. Valid values: 0(heuristic), 1(autotuning), 2(hybrid)", 
                             FLAGS_FC_TUNING_MODE));

    std::vector<MoeSortingExCodeGen> all_instances;

    // Configuration loading based on enabled flags
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / "moe" / "moe_sorting_ex";

    // Load backup configurations (pre-validated single configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        try {
            std::filesystem::path backup_path = base_json_path / "backup_config.json";
            if (std::filesystem::exists(backup_path)) {
                auto backup_configs = LoadConfigJson<std::vector<MoeSortingExConfig>>(backup_path);
                for (const auto& config : backup_configs) {
                    auto backup_instances = CreateInstanceForConfig(config, moe_sorting_ex_problem);
                    all_instances.insert(all_instances.end(), backup_instances.begin(), backup_instances.end());
                }
                VLOG(2) << "Loaded " << backup_configs.size() << " MoE sorting backup configurations";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE sorting backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        try {
            std::filesystem::path default_path = base_json_path / "default_config.json";
            if (std::filesystem::exists(default_path)) {
                auto default_config = LoadConfigJson<MoeSortingExConfig>(default_path);
                auto default_instances = CreateInstanceForConfig(default_config, moe_sorting_ex_problem);
                all_instances.insert(all_instances.end(), default_instances.begin(), default_instances.end());
                VLOG(2) << "Loaded MoE sorting default configuration with " << default_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE sorting default config: " << e.what();
        }
    }

    // Load user configurations (custom parameter ranges)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        try {
            std::filesystem::path user_path = base_json_path / "user_config.json";
            if (std::filesystem::exists(user_path)) {
                auto user_config = LoadConfigJson<MoeSortingExConfig>(user_path);
                auto user_instances = CreateInstanceForConfig(user_config, moe_sorting_ex_problem);
                all_instances.insert(all_instances.end(), user_instances.begin(), user_instances.end());
                VLOG(2) << "Loaded MoE sorting user configuration with " << user_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE sorting user config: " << e.what();
        }
    }

    // Apply mode-specific processing
    std::vector<MoeSortingExCodeGen> final_instances;
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {  // Heuristic mode: filter + random selection for fast execution
            auto filtered_instances = HeuristicFilter(all_instances, moe_sorting_ex_problem);
            if (!filtered_instances.empty()) {
                // Randomly select one optimal configuration for fast execution
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, filtered_instances.size() - 1);
                final_instances.push_back(filtered_instances[dis(gen)]);
                VLOG(1) << "MoE sorting heuristic mode: selected 1 instance from " 
                        << filtered_instances.size() << " filtered candidates";
            }
            break;
        }
        case 1: {  // Autotuning mode: use all valid instances for comprehensive search
            final_instances = all_instances;
            VLOG(1) << "MoE sorting autotuning mode: using all " << all_instances.size() << " instances";
            break;
        }
        case 2: {  // Hybrid mode: heuristic filtering + broader search
            auto filtered_instances = HeuristicFilter(all_instances, moe_sorting_ex_problem);
            final_instances = filtered_instances.empty() ? all_instances : filtered_instances;
            VLOG(1) << "MoE sorting hybrid mode: using " << final_instances.size() 
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

    VLOG(1) << "Generated " << num_instances_ << " valid MoE sorting instances "
            << " (mode " << FLAGS_FC_TUNING_MODE << ")";
}

void MoeSortingExEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
