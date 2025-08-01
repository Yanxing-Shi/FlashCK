#include "core/profiling/gemm/gemm_emitter.h"

#include "core/utils/common.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for gemm operation: 0 - heuristic, 1 - autotuning, 2 - hybrid
FC_DECLARE_bool(FC_ENABLE_CONFIG_JSON);
FC_DECLARE_string(FC_CONFIG_JSON_PATH);
FC_DECLARE_int32(FC_ENABLE_JSON_MODE);

namespace flashck {

bool TopKSoftmaxEmitter::IsValidTile(const TopKSoftmaxTileDesc& tile_desc, const TopKSoftmaxProblem& gemm_problem) const
{
    // Validate tile descriptor parameters (all tile dims > 0)
    if (tile_desc.m_block_ <= 0 || tile_desc.n_block_ <= 0 || tile_desc.k_block_ <= 0
        || tile_desc.m_warp_ <= 0 || tile_desc.n_warp_ <= 0 || tile_desc.k_warp_ <= 0
        || tile_desc.m_warp_tile_ <= 0 || tile_desc.n_warp_tile_ <= 0 || tile_desc.k_warp_tile_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Validate thread block size (example: m_block_ * n_block_ should not exceed 1024)
    const int total_threads = tile_desc.m_block_ * tile_desc.n_block_;
    if (total_threads > 1024) {
        VLOG(3) << "Invalid tile descriptor: thread block size " << total_threads << " exceeds limit (1024)";
        return false;
    }

    // Validate against problem dimensions
    if (tile_desc.m_block_ > gemm_problem.m_ || tile_desc.n_block_ > gemm_problem.n_ || tile_desc.k_block_ > gemm_problem.k_) {
        VLOG(3) << "Invalid tile descriptor: block dims (" << tile_desc.m_block_ << "," << tile_desc.n_block_ << "," << tile_desc.k_block_
                << ") exceed problem dims (" << gemm_problem.m_ << "," << gemm_problem.n_ << "," << gemm_problem.k_ << ")";
        return false;
    }

    std::tuple<int, int, int> warp_tuple = std::make_tuple(tile_desc.m_warp_, tile_desc.n_warp_, tile_desc.k_warp_);
    if (std::find(g_tile_gemm_allowed_warp_combinations.begin(), g_tile_gemm_allowed_warp_combinations.end(), warp_tuple) == g_tile_gemm_allowed_warp_combinations.end()) {
        VLOG(3) << "Invalid warp combination: warp_m(" << tile_desc.m_warp_ << ") * warp_n(" << tile_desc.n_warp_ << ") * warp_k(" << tile_desc.k_warp_ << ")";
        return false;
    }

    // Dimension alignment check: tile dims must be divisible by warp*warp_tile dims
    if (tile_desc.m_block_ % (tile_desc.m_warp_ * tile_desc.m_warp_tile_) != 0) {
        VLOG(3) << "Dimension alignment failed: tile_m(" << tile_desc.m_block_ << ") % [" << tile_desc.m_warp_ << "x" << tile_desc.m_warp_tile_ << "] = "
                << (tile_desc.m_block_ % (tile_desc.m_warp_ * tile_desc.m_warp_tile_));
        return false;
    }
    if (tile_desc.n_block_ % (tile_desc.n_warp_ * tile_desc.n_warp_tile_) != 0) {
        VLOG(3) << "Dimension alignment failed: tile_n(" << tile_desc.n_block_ << ") % [" << tile_desc.n_warp_ << "x" << tile_desc.n_warp_tile_ << "] = "
                << (tile_desc.n_block_ % (tile_desc.n_warp_ * tile_desc.n_warp_tile_));
        return false;
    }
    if (tile_desc.k_block_ % (tile_desc.k_warp_ * tile_desc.k_warp_tile_) != 0) {
        VLOG(3) << "Dimension alignment failed: tile_k(" << tile_desc.k_block_ << ") % [" << tile_desc.k_warp_ << "x" << tile_desc.k_warp_tile_ << "] = "
                << (tile_desc.k_block_ % (tile_desc.k_warp_ * tile_desc.k_warp_tile_));
        return false;
    }

    // LDS capacity verification
    size_t matrix_a_size = tile_desc.m_block_ * tile_desc.k_block_ *  SizeOf(gemm_problem.a_dtype_);
    size_t matrix_b_size = tile_desc.n_block_ * tile_desc.k_block_ * SizeOf(gemm_problem.b_dtype_);
    size_t total_tile_in_lds = matrix_a_size + matrix_b_size;

    size_t max_tile_size =  (1 << 16); // 64KB

    if (total_tile_in_lds > max_tile_size) {
        VLOG(3) << "LDS capacity exceeded: Total required " << total_tile_in_lds << "B (" << (total_tile_in_lds / 1024.0) << "KB) > "
                << "maximum allowed " << max_tile_size << "B (" << (max_tile_size / 1024) << "KB). Breakdown:\n"
                << "- Matrix A (" << DataTypeToString(gemm_problem.a_dtype_) << "): " << tile_desc.m_block_ << "x" << tile_desc.k_block_ << " = " << matrix_a_size << "B\n"
                << "- Matrix B (" << DataTypeToString(gemm_problem.b_dtype_) << "): " << tile_desc.n_block_ << "x" << tile_desc.k_block_ << " = " << matrix_b_size << "B";
        return false;
    }

    // Warp tile combination validation
    // Compose warp_tile_key as "matrix_a_matrix_b_matrix_c"
    std::string warp_tile_key = Sprintf("{}_{}_{}", DataTypeToString(gemm_problem.a_dtype_),
                                         DataTypeToString(gemm_problem.b_dtype_),
                                         DataTypeToString(gemm_problem.c_dtype_));
    std::array<int64_t, 3> current_combination = {tile_desc.m_warp_tile_, tile_desc.n_warp_tile_, tile_desc.k_warp_tile_};

    std::string gpu_name = GetDeviceName();

    auto gpu_warp_tile_key_it = g_tile_gemm_warp_tile_supported_combinations.find(gpu_name);
    if (gpu_warp_tile_key_it == g_tile_gemm_warp_tile_supported_combinations.end()) {
        VLOG(3) << "Trait: [GEMM], No valid warp tile combinations found for " << gpu_name << "/" << warp_tile_key << ", skip this check.";
        return false;
    }
    const auto& gpu_warp_tile_key = gpu_warp_tile_key_it->second;
    auto allowed_combinations_it = gpu_warp_tile_key.find(warp_tile_key);
    if (allowed_combinations_it == gpu_warp_tile_key.end()) {
        VLOG(3) << "Trait: [GEMM], No valid warp tile combinations found for " << gpu_name << "/" << warp_tile_key << ", skip this check.";
        return false;
    }
    const auto& allowed_combinations = allowed_combinations_it->second;
    if (std::find(allowed_combinations.begin(), allowed_combinations.end(), current_combination) == allowed_combinations.end()) {
        VLOG(3) << "Trait: [GEMM], Invalid warp combination: [" << current_combination[0] << ", " << current_combination[1] << ", " << current_combination[2]
                << "] not in allowed list. Valid combinations for data type '" << warp_tile_key << "': ";
        for (const auto& comb : allowed_combinations) {
            VLOG(3) << "  [" << comb[0] << ", " << comb[1] << ", " << comb[2] << "]";
        }
        return false;
    }

    return true;
}

bool TopKSoftmaxEmitter::IsValidCombination(const PipelineVersionEnum& pipeline, const EpilogueEnum& epilogue, const PipelineSchedulerEnum& scheduler)
{
    // Check if the current combination is valid (compare enums, not strings)
    return std::find(g_tile_gemm_unsupported_combinations.begin(), g_tile_gemm_unsupported_combinations.end(),
                    std::make_tuple(pipeline, epilogue, scheduler)) == g_tile_gemm_unsupported_combinations.end();
}

bool TopKSoftmaxEmitter::IsValidInstance(const TopKSoftmaxCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_) && IsValidCombination(instance.pipeline_version_, instance.pipeline_epilogue_, instance.pipeline_scheduler_);
}

// std::vector<TopKSoftmaxTileDesc> TopKSoftmaxEmitter::HeuristicFilter(const std::vector<TopKSoftmaxTileDesc>& gemm_tile_desc,
//                                                        const TopKSoftmaxProblem&               gemm_problem) const
// {
    
// }


// Generate all possible TopKSoftmaxCodeGen instances from a TopKSoftmaxConfig
std::vector<TopKSoftmaxCodeGen> TopKSoftmaxEmitter::CreateInstanceForConfig(const TopKSoftmaxConfig& config, const TopKSoftmaxProblem& gemm_problem) {
    std::vector<TopKSoftmaxCodeGen> result;

    std::vector<std::vector<int64_t>> all_lists = {
        // BlockConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.block.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.block.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.block.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.warp.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.warp.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.warp.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpTileConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.warp_tile.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.warp_tile.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile.warp_tile.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.min_block_per_cu.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PartitionConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.partition.num_wave_groups.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.partition.tile_partitioner_group_num.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.partition.tile_partitioner_m01.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PipelineConfig (enum as int64_t)
        [&]{ std::vector<int64_t> v; for (const auto& x : config.pipeline.version.values) v.emplace_back(static_cast<int64_t>(GetPipelineVersionEnumFromString(x))); return v; }(),
        [&]{ std::vector<int64_t> v; for (const auto& x : config.pipeline.scheduler.values) v.emplace_back(static_cast<int64_t>(GetPipelineSchedulerEnumFromString(x))); return v; }(),
        [&]{ std::vector<int64_t> v; for (const auto& x : config.pipeline.epilogue.values) v.emplace_back(static_cast<int64_t>(GetEpilogueEnumFromString(x))); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t m_block = vals[idx++];
        int64_t n_block = vals[idx++];
        int64_t k_block = vals[idx++];
        // WarpConfig
        int64_t m_warp = vals[idx++];
        int64_t n_warp = vals[idx++];
        int64_t k_warp = vals[idx++];
        // WarpTileConfig
        int64_t m_warp_tile = vals[idx++];
        int64_t n_warp_tile = vals[idx++];
        int64_t k_warp_tile = vals[idx++];
        // PaddingConfig
        bool pad_m = static_cast<bool>(vals[idx++]);
        bool pad_n = static_cast<bool>(vals[idx++]);
        bool pad_k = static_cast<bool>(vals[idx++]);
        // LaunchConfig
        int64_t min_block_per_cu = vals[idx++];
        // PartitionConfig
        int64_t num_wave_groups = vals[idx++];
        int64_t tile_partitioner_group_num = vals[idx++];
        int64_t tile_partitioner_m01 = vals[idx++];
        // PipelineConfig
        auto version = static_cast<PipelineVersionEnum>(vals[idx++]);
        auto scheduler = static_cast<PipelineSchedulerEnum>(vals[idx++]);
        // EpilogueConfig
        auto epilogue = static_cast<EpilogueEnum>(vals[idx++]);

        // Construct  TopKSoftmaxCodeGen
        TopKSoftmaxCodeGen gemm;
        gemm.problem_ = gemm_problem;
        gemm.tile_desc_ = TopKSoftmaxTileDesc{m_block, n_block, k_block, m_warp, n_warp, k_warp, m_warp_tile, n_warp_tile, k_warp_tile};
        gemm.pipeline_version_ = version;
        gemm.pipeline_scheduler_ = scheduler;
        gemm.pipeline_epilogue_ = epilogue;
        gemm.is_pad_m_ = pad_m;
        gemm.is_pad_n_ = pad_n;
        gemm.is_pad_k_ = pad_k;
        gemm.min_block_per_cu_ = min_block_per_cu;
        gemm.num_wave_groups_ = num_wave_groups;
        gemm.tile_partitioner_group_num_ = tile_partitioner_group_num;
        gemm.tile_partitioner_m01_ = tile_partitioner_m01;
        result.push_back(gemm);
    });
    return result;
}

void TopKSoftmaxEmitter::GenerateInstances(TopKSoftmaxProblem& gemm_problem)
{
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE == 0 || FLAGS_FC_TUNING_MODE == 1 || FLAGS_FC_TUNING_MODE == 2,
                  true,
                  Unavailable("Unsupported mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", FLAGS_FC_TUNING_MODE));

    // Check if instances already exist for this GEMM kind
    if (instance_map_.find(gemm_problem.kind_) != instance_map_.end() && !instance_map_[gemm_problem.kind_].empty()) {
        VLOG(2) << "Instances already generated for GEMM kind: " << GetTopKSoftmaxKindName(gemm_problem.kind_);
        return;
    }

    // Load tile GEMM configuration if available
    std::vector<TopKSoftmaxCodeGen> gemm_instances;
    if (FLAGS_FC_ENABLE_CONFIG_JSON) {
        std::filesystem::path base_json_path = FLAGS_FC_CONFIG_JSON_PATH;
        if(FLAGS_FC_ENABLE_JSON_MODE == 0) {
            std::filesystem::path json_path = base_json_path / "default_config.json";
            TopKSoftmaxConfig config = LoadConfigJson<TopKSoftmaxConfig>(json_path);
            gemm_instances = CreateInstanceForConfig(config, gemm_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 1) {
            std::filesystem::path json_path = base_json_path / "user_config.json";
            TopKSoftmaxConfig config = LoadConfigJson<TopKSoftmaxConfig>(json_path);
            gemm_instances = CreateInstanceForConfig(config, gemm_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 2) {
            std::filesystem::path default_json_path = base_json_path / "default_config.json";
            TopKSoftmaxConfig default_config = LoadConfigJson<TopKSoftmaxConfig>(default_json_path);
            auto gemm_default_instances = CreateInstanceForConfig(default_config, gemm_problem);

            std::filesystem::path user_json_path = base_json_path / "user_config.json";
            TopKSoftmaxConfig user_config = LoadConfigJson<TopKSoftmaxConfig>(user_json_path);
            auto gemm_user_instances = CreateInstanceForConfig(user_config, gemm_problem);

            gemm_instances.insert(gemm_instances.end(), gemm_default_instances.begin(), gemm_default_instances.end());
            gemm_instances.insert(gemm_instances.end(), gemm_user_instances.begin(), gemm_user_instances.end());
        } else{
            LOG(WARNING)<< "FC_ENABLE_JSON_MODE is set to an unsupported value: " << FLAGS_FC_ENABLE_JSON_MODE;
        }
    } else{
        LOG(WARNING)<< "FC_ENABLE_CONFIG_JSON is not enabled";
    }

    for (const auto& config : g_tile_gemm_backup_tile_config) {
        TopKSoftmaxCodeGen gemm;

        gemm.problem_ = gemm_problem;

        gemm.tile_desc_ = TopKSoftmaxTileDesc{
            config.tile.block.m.values[0],
            config.tile.block.n.values[0],
            config.tile.block.k.values[0],
            config.tile.warp.m.values[0],
            config.tile.warp.n.values[0],
            config.tile.warp.k.values[0],
            config.tile.warp_tile.m.values[0],
            config.tile.warp_tile.n.values[0],
            config.tile.warp_tile.k.values[0],
        };

        gemm.pipeline_version_ = GetPipelineVersionEnumFromString(config.pipeline.version.values[0]);
        gemm.pipeline_scheduler_ = GetPipelineSchedulerEnumFromString(config.pipeline.scheduler.values[0]);
        gemm.pipeline_epilogue_ = GetEpilogueEnumFromString(config.pipeline.epilogue.values[0]);

        gemm.min_block_per_cu_ = config.launch.min_block_per_cu.values[0];
        gemm.num_wave_groups_ = config.partition.num_wave_groups.values[0];
        gemm.tile_partitioner_group_num_ = config.partition.tile_partitioner_group_num.values[0];
        gemm.tile_partitioner_m01_ = config.partition.tile_partitioner_m01.values[0];

        gemm_instances.push_back(gemm);
    }

    // check instances
    std::vector<TopKSoftmaxCodeGen> valid_gemm_instances;
    for (const auto& gemm_instance : gemm_instances) {
        if (IsValidInstance(gemm_instance)) {
            valid_gemm_instances.push_back(gemm_instance);
        }
    }

    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {
            // 
        }
        case 1: {
            VLOG(1) << "Generating instances using autotuning mode for GEMM kind: "
                    << GetTopKSoftmaxKindName(gemm_problem.kind_);
            break;
        }
        case 2: {
            VLOG(1) << "Generating instances using hybrid mode for GEMM kind: " << GetTopKSoftmaxKindName(gemm_problem.kind_);
            break;
        }
        default:
            FC_THROW(Unavailable("Invalid mode:{} ", FLAGS_FC_TUNING_MODE));
    }


    if (valid_gemm_instances.empty()) {
        FC_THROW(Unavailable("No valid GEMM instances found for GEMM problem"));
    }
    

    // Generate instances
    std::map<std::string, TopKSoftmaxCodeGen>& kind_instance_map = instance_map_[gemm_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& instance : valid_gemm_instances) {
        try {
            std::string instance_name = instance.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = std::move(instance);
                generated_count++;
                VLOG(2) << "Generated GEMM instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate GEMM instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create GEMM codegen for instance: " << instance.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " GEMM instances for kind: " << GetTopKSoftmaxKindName(gemm_problem.kind_)
            << " (total: " << num_instances_ << ")";
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