#include "core/profiling/tile/fmha/fmha_fwd_append_kv/fmha_fwd_append_kv_emitter.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for FMHA operation: 0 - heuristic, 1 - autotuning, 2 - hybrid
FC_DECLARE_bool(FC_ENABLE_CONFIG_JSON);
FC_DECLARE_string(FC_CONFIG_JSON_PATH);
FC_DECLARE_int32(FC_ENABLE_JSON_MODE);

namespace flashck {

bool FmhaFwdAppendKVEmitter::IsValidTile(const FmhaFwdAppendKVTileDesc& tile_desc, const FmhaProblem& fmha_problem)
{


    return true;
}

bool FmhaFwdAppendKVEmitter::IsValidInstance(const FmhaFwdAppendKVCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
} 


// std::vector<FmhaFwdAppendKVTileDesc> FmhaFwdAppendKVEmitter::HeuristicFilter(const std::vector<FmhaFwdAppendKVTileDesc>& fmha_tile_desc,
//                                                        const FmhaProblem&               fmha_problem) const
// {
// }

// std::vector<FmhaFwdAppendKVAppendKVTileDesc> FmhaFwdAppendKVEmitter::HeuristicFilter(const std::vector<FmhaFwdAppendKVAppendKVTileDesc>& fmha_tile_desc,
//                                                                const FmhaProblem& fmha_problem) const
// {   
// }

// std::vector<FmhaFwdAppendKVSplitKVCombineTileDesc>
// FmhaFwdAppendKVEmitter::HeuristicFilter(const std::vector<FmhaFwdAppendKVSplitKVCombineTileDesc>& fmha_tile_desc,
//                              const FmhaProblem&                             fmha_problem) const
// {
// }


// Generate all possible FmhaFwdAppendKVCodeGen instances from a FmhaFwdAppendKVConfig
std::vector<FmhaFwdAppendKVCodeGen> FmhaFwdAppendKVEmitter::CreateInstanceForConfig(const FmhaFwdAppendKVConfig& config, const FmhaProblem& fmha_problem) {
    std::vector<FmhaFwdAppendKVCodeGen> result;

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
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.s.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.sk.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.d.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.tile.block.dv.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.s.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.sk.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.d.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.padding.dv.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<ProductElem> v; for (auto x : flatten(config.launch.min_block_per_cu.values)) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<flashck::ProductElem>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t s_block = std::get<int64_t>(vals[idx++]);
        int64_t sk_block = std::get<int64_t>(vals[idx++]);
        int64_t d_block = std::get<int64_t>(vals[idx++]);
        int64_t dv_block = std::get<int64_t>(vals[idx++]);

        // PaddingConfig
        bool is_pad_q_seq_len = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_kv_seq_len = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_qk_head_dim = static_cast<bool>(std::get<int64_t>(vals[idx++]));
        bool is_pad_v_head_dim = static_cast<bool>(std::get<int64_t>(vals[idx++]));

        // launch config
        int64_t min_block_per_cu = std::get<int64_t>(vals[idx++]);

        // Construct FmhaFwdAppendKVCodeGen
        FmhaFwdAppendKVCodeGen fmha;
        fmha.problem_ = fmha_problem;
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
        fmha.min_block_per_cu_ = min_block_per_cu;
        result.push_back(fmha);
    });

    return result;
}

void FmhaFwdAppendKVEmitter::GenerateInstances(FmhaProblem& fmha_problem)
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
    std::vector<FmhaFwdAppendKVCodeGen> fmha_instances;
    if (FLAGS_FC_ENABLE_CONFIG_JSON) {
        auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetFmhaKindName(fmha_problem.kind_);
        if(FLAGS_FC_ENABLE_JSON_MODE == 0) {
            std::filesystem::path json_path = base_json_path / "default_config.json";
            FmhaFwdAppendKVConfig config = LoadConfigJson<FmhaFwdAppendKVConfig>(json_path);
            fmha_instances = CreateInstanceForConfig(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 1) {
            std::filesystem::path json_path = base_json_path / "user_config.json";
            FmhaFwdAppendKVConfig config = LoadConfigJson<FmhaFwdAppendKVConfig>(json_path);
            fmha_instances = CreateInstanceForConfig(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 2) {
            std::filesystem::path default_json_path = base_json_path / "default_config.json";
            FmhaFwdAppendKVConfig default_config = LoadConfigJson<FmhaFwdAppendKVConfig>(default_json_path);
            auto gemm_default_instances = CreateInstanceForConfig(default_config, fmha_problem);

            std::filesystem::path user_json_path = base_json_path / "user_config.json";
            FmhaFwdAppendKVConfig user_config = LoadConfigJson<FmhaFwdAppendKVConfig>(user_json_path);
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

    for (const auto& config : g_backup_fmha_append_kv_config) {
        FmhaFwdAppendKVCodeGen fmha;

        fmha.problem_ = fmha_problem;

        fmha.tile_desc_ = FmhaFwdAppendKVTileDesc{
            config.tile.block.s.values[0][0],
            config.tile.block.sk.values[0][0],
            config.tile.block.d.values[0][0],
            config.tile.block.dv.values[0][0]
        };

        fmha.is_pad_q_seq_len_ = config.padding.s.values[0];
        fmha.is_pad_kv_seq_len_ = config.padding.sk.values[0];
        fmha.is_pad_qk_head_dim_ = config.padding.d.values[0];
        fmha.is_pad_v_head_dim_ = config.padding.dv.values[0];

        fmha.min_block_per_cu_ = config.launch.min_block_per_cu.values[0][0];

        fmha_instances.push_back(fmha);
    }

    // check instances
    std::vector<FmhaFwdAppendKVCodeGen> valid_fmha_instances;
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

void FmhaFwdAppendKVEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
