#include "core/profiling/moe/topk_softmax/topk_softmax_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files

namespace flashck {

bool TopKSoftmaxEmitter::IsValidInstance(const TopKSoftmaxCodeGen& instance)
{
    const auto& problem = instance.problem_;
    
    // Validate parameters are positive
    if (instance.num_experts_ <= 0 || instance.issues_pre_col_ <= 0 || 
        instance.bytes_per_issue_ <= 0 || instance.block_size_ <= 0 ||
        instance.min_block_pre_cu_ <= 0) {
        VLOG(3) << "Invalid TopK Softmax instance: negative or zero parameters not allowed";
        return false;
    }

    // Validate block size doesn't exceed hardware limits
    if (instance.block_size_ > 1024) {
        VLOG(3) << "Invalid TopK Softmax instance: block size " << instance.block_size_ 
                << " exceeds hardware limit (1024)";
        return false;
    }

    // Validate memory access pattern efficiency
    const size_t memory_per_token = instance.num_experts_ * SizeOf(problem.input_dtype_);
    const size_t total_memory = problem.num_tokens_ * memory_per_token;
    const size_t max_memory = 1ULL << 31; // 2GB limit
    if (total_memory > max_memory) {
        VLOG(3) << "Invalid TopK Softmax instance: estimated memory " << total_memory 
                << " exceeds limit " << max_memory;
        return false;
    }

    return true;
}

std::vector<TopKSoftmaxCodeGen> TopKSoftmaxEmitter::HeuristicFilter(
    const std::vector<TopKSoftmaxCodeGen>& instances,
    const MoeProblem& moe_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<TopKSoftmaxCodeGen> filtered;
    
    // Score and rank instances based on TopK Softmax performance heuristics
    std::vector<std::pair<double, size_t>> scored_instances;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& instance = instances[i];
        double score = 0.0;
        
        // 1. Memory bandwidth efficiency (prefer configurations that maximize throughput)
        if (instance.issues_pre_col_ >= 2 && instance.issues_pre_col_ <= 8) {
            score += 0.3;  // Good memory pipeline utilization
        }
        
        // 2. Block size efficiency for parallel reduction
        if (instance.block_size_ >= 256 && instance.block_size_ <= 512) {
            score += 0.25;  // Sweet spot for reduction operations
        }
        
        // 3. TopK selectivity optimization
        const double selectivity = static_cast<double>(moe_problem.topk_) / instance.num_experts_;
        if (selectivity >= 0.1 && selectivity <= 0.5) {
            score += 0.15;  // Good sparsity balance
        }
        
        // 4. Memory access pattern optimization
        if (instance.bytes_per_issue_ == 4 || instance.bytes_per_issue_ == 8) {
            score += 0.1;  // Aligned memory access
        }
        
        scored_instances.emplace_back(score, i);
    }
    
    // Sort by score (highest first)
    std::sort(scored_instances.begin(), scored_instances.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top candidates (limit to reasonable number for heuristic mode)
    size_t max_candidates = std::min(static_cast<size_t>(12), instances.size());
    filtered.reserve(max_candidates);
    
    for (size_t i = 0; i < max_candidates; ++i) {
        filtered.push_back(instances[scored_instances[i].second]);
    }
    
    VLOG(2) << "TopK Softmax heuristic filter: reduced " << instances.size() 
            << " instances to " << filtered.size() << " candidates";
    
    return filtered;
}

std::vector<TopKSoftmaxCodeGen> TopKSoftmaxEmitter::CreateInstanceForConfig(
    const TopKSoftmaxConfig& config, const MoeProblem& moe_problem) 
{
    std::vector<TopKSoftmaxCodeGen> result;

    // Convert all config parameters to int64_t vectors for CartesianProduct
    std::vector<std::vector<int64_t>> all_param_lists = {
        // Memory access configuration
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.issues_per_col.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.bytes_per_issue.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        // Launch configuration
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch_type.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.block_size.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.min_block_per_cu.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
    };

    CartesianProduct(all_param_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        
        // Memory access configuration
        int issues_per_col = static_cast<int>(vals[idx++]);
        int bytes_per_issue = static_cast<int>(vals[idx++]);
        
        // Launch configuration
        int launch_type = static_cast<int>(vals[idx++]);
        int64_t block_size = vals[idx++];
        int min_block_per_cu = static_cast<int>(vals[idx++]);

        // Construct TopKSoftmaxCodeGen
        TopKSoftmaxCodeGen instance;
        instance.problem_ = moe_problem;
        instance.issues_pre_col_ = issues_per_col;
        instance.bytes_per_issue_ = bytes_per_issue;
        instance.launch_type_ = launch_type;
        instance.block_size_ = block_size;
        instance.min_block_pre_cu_ = min_block_per_cu;
        
        result.push_back(instance);
    });
    
    return result;
}

void TopKSoftmaxEmitter::GenerateInstances(MoeProblem& moe_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid FC_TUNING_MODE: {}. Valid values: 0(heuristic), 1(autotuning), 2(hybrid)", 
                             FLAGS_FC_TUNING_MODE));

    // Check if instances already exist for this TopK Softmax kind
    if (instance_map_.find(moe_problem.kind_) != instance_map_.end() && 
        !instance_map_[moe_problem.kind_].empty()) {
        VLOG(2) << "TopK Softmax instances already generated for kind: " << GetTopKSoftmaxKindName(moe_problem.kind_);
        return;
    }

    std::vector<TopKSoftmaxCodeGen> all_instances;

    // Configuration loading based on enabled flags
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / "topk_softmax";

    // Load backup configurations (pre-validated single configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        try {
            std::filesystem::path backup_path = base_json_path / "backup_config.json";
            if (std::filesystem::exists(backup_path)) {
                auto backup_configs = LoadConfigJson<std::vector<TopKSoftmaxConfig>>(backup_path);
                for (const auto& config : backup_configs) {
                    auto backup_instances = CreateInstanceForConfig(config, moe_problem);
                    all_instances.insert(all_instances.end(), backup_instances.begin(), backup_instances.end());
                }
                VLOG(2) << "Loaded " << backup_configs.size() << " TopK Softmax backup configurations";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load TopK Softmax backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        try {
            std::filesystem::path default_path = base_json_path / "default_config.json";
            if (std::filesystem::exists(default_path)) {
                auto default_config = LoadConfigJson<TopKSoftmaxConfig>(default_path);
                auto default_instances = CreateInstanceForConfig(default_config, moe_problem);
                all_instances.insert(all_instances.end(), default_instances.begin(), default_instances.end());
                VLOG(2) << "Loaded TopK Softmax default configuration with " << default_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load TopK Softmax default config: " << e.what();
        }
    }

    // Load user configurations (custom parameter ranges)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        try {
            std::filesystem::path user_path = base_json_path / "user_config.json";
            if (std::filesystem::exists(user_path)) {
                auto user_config = LoadConfigJson<TopKSoftmaxConfig>(user_path);
                auto user_instances = CreateInstanceForConfig(user_config, moe_problem);
                all_instances.insert(all_instances.end(), user_instances.begin(), user_instances.end());
                VLOG(2) << "Loaded TopK Softmax user configuration with " << user_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load TopK Softmax user config: " << e.what();
        }
    }

    // Apply mode-specific processing
    std::vector<TopKSoftmaxCodeGen> final_instances;
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {  // Heuristic mode: filter + random selection for fast execution
            auto filtered_instances = HeuristicFilter(all_instances, moe_problem);
            if (!filtered_instances.empty()) {
                // Randomly select one optimal configuration for fast execution
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, filtered_instances.size() - 1);
                final_instances.push_back(filtered_instances[dis(gen)]);
                VLOG(1) << "TopK Softmax heuristic mode: selected 1 instance from " 
                        << filtered_instances.size() << " filtered candidates";
            }
            break;
        }
        case 1: {  // Autotuning mode: use all valid instances for comprehensive search
            final_instances = all_instances;
            VLOG(1) << "TopK Softmax autotuning mode: using all " << all_instances.size() << " instances";
            break;
        }
        case 2: {  // Hybrid mode: heuristic filtering + broader search
            auto filtered_instances = HeuristicFilter(all_instances, moe_problem);
            final_instances = filtered_instances.empty() ? all_instances : filtered_instances;
            VLOG(1) << "TopK Softmax hybrid mode: using " << final_instances.size() 
                    << " instances (filtered from " << all_instances.size() << ")";
            break;
        }
    }

    // Validate and store instances
    num_instances_ = 0;
    for (const auto& instance : final_instances) {
        if (IsValidInstance(instance)) {
            instance_map_[moe_problem.kind_][instance.GetInstanceName()] = instance;
            ++num_instances_;
        }
    }

    VLOG(1) << "Generated " << num_instances_ << " valid TopK Softmax instances for " 
            << GetTopKSoftmaxKindName(moe_problem.kind_) << " (mode " << FLAGS_FC_TUNING_MODE << ")";
}

int64_t TopKSoftmaxEmitter::GetNumInstances() const
{
    return num_instances_;
}

void TopKSoftmaxEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck

