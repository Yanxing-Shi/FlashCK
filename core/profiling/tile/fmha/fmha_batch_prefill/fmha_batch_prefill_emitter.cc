#include "core/profiling/tile/fmha/fmha_batch_prefill/fmha_batch_prefill_emitter.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for FMHA operation: 0 - heuristic, 1 - autotuning, 2 - hybrid
FC_DECLARE_bool(FC_ENABLE_CONFIG_JSON);
FC_DECLARE_string(FC_CONFIG_JSON_PATH);
FC_DECLARE_int32(FC_ENABLE_JSON_MODE);

namespace flashck {

bool FmhaBatchPrefillEmitter::IsValidTile(const FmhaBatchPrefillTileDesc& tile_desc, const FmhaProblem& fmha_problem)
{
    // All tile parameters must be positive
    if (tile_desc.m0_block_ <= 0 || tile_desc.n0_block_ <= 0 || tile_desc.k0_block_ <= 0 || tile_desc.k0_max_block_ <= 0 ||
        tile_desc.n1_block_ <= 0 || tile_desc.k1_block_ <= 0 ||
        tile_desc.m0_warp_ <= 0 || tile_desc.n0_warp_ <= 0 || tile_desc.k0_warp_ < 0 ||
        tile_desc.m1_warp_ <= 0 || tile_desc.n1_warp_ <= 0 || tile_desc.k1_warp_ < 0 ||
        tile_desc.m0_warp_tile_ <= 0 || tile_desc.n0_warp_tile_ <= 0 || tile_desc.k0_warp_tile_ <= 0 ||
        tile_desc.m1_warp_tile_ <= 0 || tile_desc.n1_warp_tile_ <= 0 || tile_desc.k1_warp_tile_ <= 0) {
        VLOG(3) << "Invalid FMHA tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Cross-parameter constraints
    // 1. k0_block_ should not exceed k0_max_block_
    if (tile_desc.k0_block_ > tile_desc.k0_max_block_) {
        VLOG(3) << "Invalid FMHA tile descriptor: k0_block_ > k0_max_block_";
        return false;
    }
    // 2. Warp and warp tile sizes should not exceed block sizes
    if (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n0_warp_ * tile_desc.n0_warp_tile_ > tile_desc.n0_block_ ||
        tile_desc.k0_warp_ * tile_desc.k0_warp_tile_ > tile_desc.k0_block_ ||
        tile_desc.m1_warp_ * tile_desc.m1_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n1_warp_ * tile_desc.n1_warp_tile_ > tile_desc.n1_block_ ||
        tile_desc.k1_warp_ * tile_desc.k1_warp_tile_ > tile_desc.k1_block_) {
        VLOG(3) << "Invalid FMHA tile descriptor: warp*warp_tile exceeds block size";
        return false;
    }
    // 3. All block sizes should be divisible by warp*warp_tile sizes
    if ((tile_desc.m0_block_ % (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_) != 0) ||
        (tile_desc.n0_block_ % (tile_desc.n0_warp_ * tile_desc.n0_warp_tile_) != 0) ||
        (tile_desc.k0_block_ % (std::max<int64_t>(1, tile_desc.k0_warp_ * tile_desc.k0_warp_tile_)) != 0) ||
        (tile_desc.m0_block_ % (tile_desc.m1_warp_ * tile_desc.m1_warp_tile_) != 0) ||
        (tile_desc.n1_block_ % (tile_desc.n1_warp_ * tile_desc.n1_warp_tile_) != 0) ||
        (tile_desc.k1_block_ % (std::max<int64_t>(1, tile_desc.k1_warp_ * tile_desc.k1_warp_tile_)) != 0)) {
        VLOG(3) << "Invalid FMHA tile descriptor: block size not divisible by warp*warp_tile";
        return false;
    }

    // Validate against problem dimensions for Batch mode
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.m0_block_ > fmha_problem.q_seq_len_ || tile_desc.n0_block_ > fmha_problem.kv_seq_len_ ||
            tile_desc.n1_block_ > fmha_problem.v_head_dim_ || tile_desc.k0_max_block_ > fmha_problem.qk_head_dim_) {
            VLOG(3) << "Invalid FMHA tile descriptor: tile dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaBatchPrefillEmitter::IsValidInstance(const FmhaBatchPrefillCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
} 


// std::vector<FmhaBatchPrefillTileDesc> FmhaBatchPrefillEmitter::HeuristicFilter(const std::vector<FmhaBatchPrefillTileDesc>& fmha_tile_desc,
//                                                        const FmhaProblem&               fmha_problem) const
// {
// }

// std::vector<FmhaBatchPrefillAppendKVTileDesc> FmhaBatchPrefillEmitter::HeuristicFilter(const std::vector<FmhaBatchPrefillAppendKVTileDesc>& fmha_tile_desc,
//                                                                const FmhaProblem& fmha_problem) const
// {   
// }

// std::vector<FmhaBatchPrefillSplitKVCombineTileDesc>
// FmhaBatchPrefillEmitter::HeuristicFilter(const std::vector<FmhaBatchPrefillSplitKVCombineTileDesc>& fmha_tile_desc,
//                              const FmhaProblem&                             fmha_problem) const
// {
// }


// Generate all possible FmhaBatchPrefillCodeGen instances from a FmhaBatchPrefillConfig
std::vector<FmhaBatchPrefillCodeGen> FmhaBatchPrefillEmitter::CreateInstanceForConfig(const FmhaBatchPrefillConfig& config, const FmhaProblem& fmha_problem) {
    std::vector<FmhaBatchPrefillCodeGen> result;

    // Helper to flatten vector<vector<T>> or just return vector<T> as is
    auto flatten = [](const auto& v) {
        // Specialize for StrEnumConfigParam and similar types
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, StrEnumConfigParam>) {
            return v.values_;
        } else {
            using VecT = std::decay_t<decltype(v)>;
            using ElemT = typename VecT::value_type;
            if constexpr (std::is_same_v<ElemT, bool> || std::is_arithmetic_v<ElemT> || std::is_same_v<ElemT, std::string>) {
                // v is vector<T>, just return as is
                return std::vector<ElemT>(v.begin(), v.end());
            } else {
                // v is vector<vector<T>>
                std::vector<typename ElemT::value_type> out;
                for (const auto& inner : v) out.insert(out.end(), inner.begin(), inner.end());
                return out;
            }
        }
    };

    using ProductElem = std::variant<int64_t, std::vector<int64_t>>;
    std::vector<std::vector<ProductElem>> all_lists = {
        // BlockConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.m0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.n0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.k0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.k0_max.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.n1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.k1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp.m0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp.n0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp.k0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
         [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp.m1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp.n1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp.k1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpTileConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp_tile.m0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp_tile.n0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp_tile.k0.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp_tile.m1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp_tile.n1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.warp_tile.k1.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.s.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.sk.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.d.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.dv.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.launch.min_block_per_cu.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PipelineConfig (enum as int64_t)
        [&]{ std::vector<ProductElem> v; for (const auto& x : flatten(config.pipeline.values)) v.emplace_back(static_cast<int64_t>(GetBlockFmhaPipelineEnumFromString(x))); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<flashck::ProductElem>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t m0_block = std::get<int64_t>(vals[idx++]);
        int64_t n0_block = std::get<int64_t>(vals[idx++]);
        int64_t k0_block = std::get<int64_t>(vals[idx++]);
        int64_t n1_block = std::get<int64_t>(vals[idx++]);
        int64_t k1_block = std::get<int64_t>(vals[idx++]);
        int64_t k0_max_block = std::get<int64_t>(vals[idx++]);

        int64_t m0_warp = std::get<int64_t>(vals[idx++]);
        int64_t n0_warp = std::get<int64_t>(vals[idx++]);
        int64_t k0_warp = std::get<int64_t>(vals[idx++]);
        int64_t m1_warp = std::get<int64_t>(vals[idx++]);
        int64_t n1_warp = std::get<int64_t>(vals[idx++]);
        int64_t k1_warp = std::get<int64_t>(vals[idx++]);

        int64_t m0_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t n0_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t k0_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t m1_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t n1_warp_tile = std::get<int64_t>(vals[idx++]);
        int64_t k1_warp_tile = std::get<int64_t>(vals[idx++]);

        // PaddingConfig
        bool is_pad_q_seq_len = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_kv_seq_len = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_qk_head_dim = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_v_head_dim = static_cast<bool>(std::get<int64_t>(vals[idx++]));

        // launch config
        int64_t min_block_per_cu = std::get<int64_t>(vals[idx++]);

        // PipelineConfig
        BlockFmhaPipelineEnum pipeline = static_cast<BlockFmhaPipelineEnum>(std::get<int64_t>(vals[idx++]));

        // Construct FmhaBatchPrefillCodeGen
        FmhaBatchPrefillCodeGen fmha;
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
        fmha.is_pad_q_seq_len_ = is_pad_q_seq_len;
        fmha.is_pad_kv_seq_len_ = is_pad_kv_seq_len;
        fmha.is_pad_qk_head_dim_ = is_pad_qk_head_dim;
        fmha.is_pad_v_head_dim_ = is_pad_v_head_dim;
        // Launch
        fmha.min_block_per_cu_ = min_block_per_cu;
        // Pipeline
        fmha.pipeline_ = pipeline;
        result.push_back(fmha);
    });

    return result;
}

void FmhaBatchPrefillEmitter::GenerateInstances(FmhaProblem& fmha_problem)
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
    std::vector<FmhaBatchPrefillCodeGen> fmha_instances;
    if (FLAGS_FC_ENABLE_CONFIG_JSON) {
        auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetFmhaKindName(fmha_problem.kind_);
        if(FLAGS_FC_ENABLE_JSON_MODE == 0) {
            std::filesystem::path json_path = base_json_path / "default_config.json";
            FmhaBatchPrefillConfig config = LoadConfigJson<FmhaBatchPrefillConfig>(json_path);
            fmha_instances = CreateInstanceForConfig(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 1) {
            std::filesystem::path json_path = base_json_path / "user_config.json";
            FmhaBatchPrefillConfig config = LoadConfigJson<FmhaBatchPrefillConfig>(json_path);
            fmha_instances = CreateInstanceForConfig(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 2) {
            std::filesystem::path default_json_path = base_json_path / "default_config.json";
            FmhaBatchPrefillConfig default_config = LoadConfigJson<FmhaBatchPrefillConfig>(default_json_path);
            auto gemm_default_instances = CreateInstanceForConfig(default_config, fmha_problem);

            std::filesystem::path user_json_path = base_json_path / "user_config.json";
            FmhaBatchPrefillConfig user_config = LoadConfigJson<FmhaBatchPrefillConfig>(user_json_path);
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

    for (const auto& config : g_backup_fmha_batch_prefill_config) {
        FmhaBatchPrefillCodeGen fmha;

        fmha.problem_ = fmha_problem;

        fmha.tile_desc_ = FmhaBatchPrefillTileDesc{
            config.tile.block.m0.values[0][0],
            config.tile.block.n0.values[0][0],
            config.tile.block.k0.values[0][0],
            config.tile.block.k0_max.values[0][0],
            config.tile.block.n1.values[0][0],
            config.tile.block.k1.values[0][0],
            config.tile.warp.m0.values[0][0],
            config.tile.warp.n0.values[0][0],
            config.tile.warp.k0.values[0][0],
            config.tile.warp.m1.values[0][0],
            config.tile.warp.n1.values[0][0],
            config.tile.warp.k1.values[0][0],
            config.tile.warp_tile.m0.values[0][0],
            config.tile.warp_tile.n0.values[0][0],
            config.tile.warp_tile.k0.values[0][0],
            config.tile.warp_tile.m1.values[0][0],
            config.tile.warp_tile.n1.values[0][0],
            config.tile.warp_tile.k1.values[0][0]
        };

        fmha.is_pad_q_seq_len_ = config.padding.s.values[0];
        fmha.is_pad_kv_seq_len_ = config.padding.sk.values[0];
        fmha.is_pad_qk_head_dim_ = config.padding.d.values[0];
        fmha.is_pad_v_head_dim_ = config.padding.dv.values[0];

        fmha.min_block_per_cu_ = config.launch.min_block_per_cu.values[0][0];

        fmha.pipeline_ = GetBlockFmhaPipelineEnumFromString(config.pipeline.values[0]);

        fmha_instances.push_back(fmha);
    }

    // check instances
    std::vector<FmhaBatchPrefillCodeGen> valid_fmha_instances;
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

void FmhaBatchPrefillEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
