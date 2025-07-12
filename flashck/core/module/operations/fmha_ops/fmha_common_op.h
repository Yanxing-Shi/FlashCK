#pragma once

FC_DECLARE_bool(FC_FORCE_PROFILE);
FC_DECLARE_bool(FC_FORCE_PROFILING_DB);
FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

template<typename CppType, typename OpType>
class FmhaCommonOp: public Operation {
public:
    FmhaCommonOp(std::string op_name): Operation(op_name) {}

    FmhaProblem DefineProblem(const std::vector<int64_t>& inverse_res)
    {
        return static_cast<OpType*>(this)->DefineProblemImpl(inverse_res);
    }

    int64_t GetBiasRankInfo(const BiasEnum& bias_enum, Variable* q, Variable* k, Variable* bias)
    {
        DDim batch_size  = q->GetShape().GetDim(0);
        DDim q_seq_len   = q->GetShape().GetDim(1);
        DDim kv_seq_len  = k->GetShape().GetDim(1);
        DDim q_num_heads = q->GetShape().GetDim(2);

        if (bias_enum == BiasEnum::ELEMENTWISE_BIAS) {
            if (bias->GetShape() == Shape{DDim(1), DDim(1), q_seq_len, kv_seq_len}) {
                return 0;
            }
            else if (bias->GetShape() == Shape{DDim(1), q_num_heads, q_seq_len, kv_seq_len}) {
                return 1;
            }
            else if (bias->GetShape() == Shape{batch_size, q_num_heads, q_seq_len, kv_seq_len}) {
                return 2;
            }
            else {
                FC_THROW(Unavailable("The shape of elementwise bias is wrong"));
            }
        }
        else if (bias_enum == BiasEnum::ALIBI) {
            if (bias->GetShape() == Shape{DDim(1), q_num_heads}) {
                return 0;
            }
            else if (bias->GetShape() == Shape{batch_size, q_num_heads}) {
                return 1;
            }
            else {
                FC_THROW(Unavailable("The shape of alibi bias is wrong"));
            }
        }
        else {
            FC_THROW(Unavailable("no bias not return rank info"));
        }
    }

    // [batch, nhead_k, nhead_q, hdim_q, hdim_v, seqlen_k, seqlen_q]
    std::vector<int64_t> InvertExecKey(const std::string& key)
    {
        std::vector<int64_t> tmp;
        std::regex           pattern("(\\d+)");
        std::smatch          m;
        std::string          s = key;
        while (std::regex_search(s, m, pattern)) {
            tmp.push_back(std::stoi(m[0]));
            s = m.suffix().str();
        }
        return tmp;
    }

    std::string GenExecKey(const std::map<std::string, std::vector<int64_t>>& name_value_mapping)
    {
        std::vector<std::string> key_strs;
        for (auto& [name, values] : name_value_mapping) {
            if (values.size() == 1) {
                key_strs.emplace_back(Sprintf("{} == {}", name, values[0]));
            }
            else if (values.size() > 1) {
                key_strs.emplace_back(Sprintf("{} >= {} && {} <= {}", name, values[0], name, values.back()));
            }
            else {
                FC_THROW(Unavailable("fmha input has empty dim values: {}", values[0]));
            }
        }

        return JoinStrings(key_strs, " && ");
    }

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDims()
    {
        return static_cast<OpType*>(this)->ExtractDimsImpl();
    }

    void ExtractExecPath(const ProfilingStrategy& dynamic_profiling_strategy = ProfilingStrategy::kMax,
                         const int                step_value                 = 1)
    {
        std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> dim_info_map = this->ExtractDims();

        // dynamic shape M:{range_lower, range_upper}, K:{range_lower, range_upper}, N:{range_lower, range_upper}
        std::map<std::string, std::vector<DDim>> dim_map;

        for (auto& [name, dim_infos] : dim_info_map) {
            std::shared_ptr<DimInfo> dim_info;
            for (auto& d : dim_infos) {
                if (d->placeholder_) {
                    continue;
                }

                if (dim_info == nullptr) {
                    dim_info = d;
                }
                else if (d->source_ == TensorSource::kInput) {
                    // input should have priority.
                    dim_info = d;
                }
            }

            if (dim_info == nullptr) {
                FC_THROW(Fatal("Couldn't find valid dim info for dim {}", name));
            }

            std::vector<Variable*> var_vec   = dim_info->source_ == TensorSource::kInput ? input_var_ : output_var_;
            std::vector<DDim>      dim_shape = var_vec[dim_info->tensor_idx_]->GetShape().ToVector();

            for (const auto& dim : dim_info->dim_idx_) {
                VLOG(1) << "name:" << name << " idx: " << dim << " value: " << dim_shape[dim].ToString();
                dim_map[name].emplace_back(dim_shape[dim]);
            }
        }

        std::map<std::string, std::vector<int64_t>> shape_values_map;

        int64_t initial_product = 1;
        for (const auto& [name, dims] : dim_map) {
            std::vector<int64_t> min_dims_value, max_dims_value;
            // dynamic shape
            for (const auto& dim : dims) {
                min_dims_value.emplace_back(dim.GetLowerBound());
                max_dims_value.emplace_back(dim.GetUpperBound());
            }

            int64_t min_value = std::accumulate(
                min_dims_value.begin(), min_dims_value.end(), initial_product, std::multiplies<int64_t>());
            int64_t max_value = std::accumulate(
                max_dims_value.begin(), max_dims_value.end(), initial_product, std::multiplies<int64_t>());

            std::vector<int64_t> shape_values{min_value, max_value};
            std::sort(shape_values.begin(), shape_values.end());

            shape_values_map[name] = shape_values;

            VLOG(1) << "name: " << name << " min_dims_value: " << min_value;
            VLOG(1) << "name: " << name << " max_dims_value: " << max_value;
        }

        if (dynamic_profiling_strategy == ProfilingStrategy::kMax) {
            std::map<std::string, std::vector<int64_t>> max_values;
            for (auto& [name, shape_values] : shape_values_map) {
                int64_t max_shape_values = *max_element(shape_values.begin(), shape_values.end());
                max_values[name]         = {max_shape_values};
            }

            VLOG(1) << "profiling_key: " << GenExecKey(max_values);
            VLOG(1) << "exec_cond: " << GenExecKey(shape_values_map);

            std::shared_ptr<RunningItem> exec_item_ptr =
                std::make_shared<RunningItem>(GenExecKey(max_values), GenExecKey(shape_values_map), "");

            exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
        }
        else if (dynamic_profiling_strategy == ProfilingStrategy::kMin) {
            std::map<std::string, std::vector<int64_t>> min_values;
            for (auto& [name, shape_values] : shape_values_map) {
                int64_t min_shape_values = *min_element(shape_values.begin(), shape_values.end());
                min_values[name]         = {min_shape_values};
            }

            std::shared_ptr<RunningItem> exec_item_ptr =
                std::make_shared<RunningItem>(GenExecKey(min_values), GenExecKey(shape_values_map), "");
            exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
        }
        else if (dynamic_profiling_strategy == ProfilingStrategy::kIteration) {
            // iteration
            std::map<std::string, std::vector<int64_t>> iter_values_map;
            size_t                                      max_value_size = 0;
            for (const auto& [name, shape_values] : shape_values_map) {
                for (int64_t i = shape_values[0]; i <= shape_values[1]; i += step_value) {
                    iter_values_map[name].push_back(i);
                }
                max_value_size = std::max(max_value_size, iter_values_map[name].size());
            }

            // if lower bound and upper bound are the same, we need to fill the vector with the same value
            for (auto& [name, values] : iter_values_map) {
                // lower bound and upper bound are the same
                if (values.size() == 1) {
                    VLOG(1) << "name: " << name << " lower bound and upper bound are the same";
                    values.resize(max_value_size);
                    std::fill(values.begin(), values.end(), values.front());
                }
                VLOG(1) << "max_step: " << max_value_size;
                VLOG(1) << "name: " << name << " iter_values: " << JoinStrings(values, ",")
                        << " size: " << values.size();
            }

            // generate exec path
            std::vector<std::map<std::string, std::vector<int64_t>>> iter_values_vec;
            for (size_t i = 0; i < max_value_size; i++) {
                std::map<std::string, std::vector<int64_t>> iter_value{};
                for (const auto& [name, values] : iter_values_map) {
                    iter_value[name].push_back(values[i]);
                }
                iter_values_vec.push_back(iter_value);
            }

            for (const auto& iter_value : iter_values_vec) {
                VLOG(1) << "profiling_key: " << GenExecKey(iter_value);
                VLOG(1) << "exec_cond: " << GenExecKey(iter_value);

                std::shared_ptr<RunningItem> exec_item_ptr =
                    std::make_shared<RunningItem>(GenExecKey(iter_value), GenExecKey(iter_value), "");
                exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
            }
        }

        else {
            FC_THROW(Unimplemented("fmha only supports MIN or MAX or Interation dynamic profiling"));
        }
    }

    void IfShouldBuildProfiler(const std::vector<std::string>& workloads)
    {
        for (const auto& workload : workloads) {
            std::string exec_entry_sha1 = SHA1ToHexString(workload);
            auto        query           = FmhaQueryEntry{DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                        g_generic_attention_mask_short_names_map.at(mask_enum_),
                                        g_bias_enum_names_map.at(bias_enum_),
                                        g_fmha_operation_mode_name_map.at(op_mode_),
                                        this->rotary_dim_,
                                        this->paged_block_size_,
                                        this->use_cache_batch_idx_,
                                        g_fmha_kind_names_map.at(op_kind_),
                                        Target::Instance()->GetTargetDeviceName(),
                                        g_short_tensor_operation_names_map.at(epilogue_op_),
                                        exec_entry_sha1};

            auto cache_value = Target::Instance()->QueryProfileCache(CodeGenKind::Fmha, query);

            if (cache_value != std::make_tuple("null", -1) && !FLAGS_FC_FORCE_PROFILE) {
                std::string best_algo = std::get<0>(cache_value);
                int64_t     split_k   = std::get<1>(cache_value);
                LOG(INFO) << "Load profiling result for" << op_name_ << "from cache, algo" << best_algo << "split_k"
                          << split_k;

                exec_path_[workload]->algo_    = best_algo;
                exec_path_[workload]->split_k_ = split_k;
            }
            else {
                // cache miss - we will have to generate and build profilers
                LOG(INFO) << "No profiling result found for" << op_name_ << "in cache, will build profilers";
            }
        }
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenOpProfiler(const ProfilingStrategy& dynamic_profiling_strategy = ProfilingStrategy::kMax) override
    {
        KernelKey kernel_key(SourceType::CK_TILE, DataLayout::ALL_LAYOUT, CppTypeToDataType<CppType>::Type());
        register_kernel_ptr_ = KernelFactory::Instance().SelectKernel(g_fmha_kind_names_map.at(op_kind_), kernel_key);

        // init exec path
        ExtractExecPath(dynamic_profiling_strategy);

        exec_key_ = GetKeyVector(exec_path_);

        if (!FLAGS_FC_FORCE_PROFILER_CACHE) {
            IfShouldBuildProfiler(exec_key_);
        }
        else {
            LOG(INFO) << "Forced to use cache, skip building profilers for " << op_name_;
            return {};
        }

        std::vector<FmhaProblem> fmha_problems;
        fmha_problems.reserve(exec_key_.size());
        std::for_each(exec_key_.begin(), exec_key_.end(), [&](const std::string& key) {
            std::vector<int64_t> inverse_res =
                InvertExecKey(key);  // [batch, nhead_k, nhead_q, hdim_q, hdim_v, seqlen_k, seqlen_q]

            fmha_problems.emplace_back(DefineProblem(inverse_res));
        });

        std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> generated_profilers;

        for (int i = 0; i < exec_key_.size(); i++) {
            Target::Instance()->GenerateKernel(CodeGenKind::Fmha, fmha_problems[i]);

            kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
            if (kernel_instance_map_.size() == 0) {
                FC_THROW(Fatal("No fmha op instances were generated for {}", op_name_));
            }

            if (exec_path_[exec_key_[i]]->algo_ == "") {
                generated_profilers = register_kernel_ptr_->GenKernelProfiler(context_ptr_->GetName(), GetAttrsMap());
            }
            else {
                LOG(INFO) << "op_name: " << op_name_ << ", " << "workload: " << exec_key_[i]
                          << " from cache, not profile";
            }
        }

        return generated_profilers;
    }

    std::vector<std::string>
    GenOpProfileCmd(const std::string&                                                 profiler_prefix,
                    const std::string&                                                 profiler_filename,
                    const std::string&                                                 exec_key,
                    const std::function<std::vector<std::string>(const std::string&)>& fbuild_cmd = nullptr)
    {
        std::filesystem::path exe_path = std::filesystem::path(profiler_prefix) / profiler_filename;

        if (!CheckExistWithRetries(exe_path, 3, 5)) {
            FC_THROW(Fatal("Profiler {} is not executable", exe_path.string()));
        }

        std::vector<std::string> cmd_args = fbuild_cmd(exec_key);

        std::vector<std::string> cmd = {exe_path.string()};

        cmd.insert(cmd.end(), cmd_args.begin(), cmd_args.end());

        return cmd;
    }

    void ProfileSingleWorkload(const std::string&                         profiler_prefix,
                               const std::string&                         workload,
                               const std::shared_ptr<GPUProfilingRunner>& profiler_runner_ptr,
                               bool                                       force_cache)
    {
        std::string exec_entry_sha1 = SHA1ToHexString(workload);
        auto        query           = FmhaQueryEntry{DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                    g_generic_attention_mask_short_names_map.at(mask_enum_),
                                    g_bias_enum_names_map.at(bias_enum_),
                                    g_fmha_operation_mode_name_map.at(op_mode_),
                                    this->rotary_dim_,
                                    this->paged_block_size_,
                                    this->use_cache_batch_idx_,
                                    g_fmha_kind_names_map.at(op_kind_),
                                    Target::Instance()->GetTargetDeviceName(),
                                    g_short_tensor_operation_names_map.at(epilogue_op_),
                                    exec_entry_sha1};

        auto cache_value = Target::Instance()->QueryProfileCache(CodeGenKind::Fmha, query);

        if (cache_value == std::make_tuple("null", -1) && force_cache) {
            LOG(WARNING) << "force_cache is enabled but we could not find the following cache available on device. "
                         << "op_name: " << op_name_ << " exec_entry_sha1: " << exec_entry_sha1;
        }

        for (const auto& kernel_instance : kernel_instance_map_) {
            auto GenCallback = [&]() {
                auto process_result_callback =
                    [&](const std::vector<ProfileResult>&           result,
                        const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr) {
                        postprocessing_delegate_ptr->AddInstance(result,
                                                                 CodeGenKind::Fmha,
                                                                 GetAttrsMap(),
                                                                 kernel_instance.first,
                                                                 workload,
                                                                 this->num_splits_);
                    };
                return process_result_callback;
            };

            auto fbuild_cmd = static_cast<OpType*>(this)->GenBuildCmd();

            std::vector<std::string> command =
                GenOpProfileCmd(profiler_prefix, kernel_instance.first, workload, fbuild_cmd);

            LOG(INFO) << "profile command: " << JoinStrings(command);

            if (op_kind_ == FmhaOperationKind::FwdSplitKV) {

                std::vector<int64_t> inverse_res =
                    InvertExecKey(workload);  // [batch, nhead_k, nhead_q, hdim_q, hdim_v, seqlen_k, seqlen_q]

                std::vector<int64_t> split_k_search_space;
                if (this->num_splits_ != -1) {
                    split_k_search_space = {this->num_splits_};
                }
                else {
                    split_k_search_space = GetSplitSearchSpace(
                        inverse_res[0],
                        inverse_res[2],
                        inverse_res[6],
                        std::static_pointer_cast<FmhaFwdSplitKVOperation>(kernel_instance.second)->tile_desc_.bm0_);
                }

                for (auto& split : split_k_search_space) {
                    command.emplace_back("-num_splits=" + std::to_string(split));
                    this->num_splits_ = split;
                    profiler_runner_ptr->Push(command, GenCallback());
                    command.pop_back();
                }
            }
            else if (op_kind_ == FmhaOperationKind::FwdSplitKVCombine) {
                profiler_runner_ptr->Push(command, GenCallback());
            }
            else {
                this->num_splits_ = -1;
                profiler_runner_ptr->Push(command, GenCallback());
            }
        }
    }

    void Profile(const std::shared_ptr<GPUProfilingRunner>& profiler_runner_ptr,
                 const std::string&                         folder_name) override
    {
        std::filesystem::path profiler_prefix =
            std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_ptr_->GetName() / "profiling" / op_name_;

        for (const auto& workload : exec_key_) {
            if (exec_path_[workload]->algo_ == "") {
                if (kernel_instance_map_.size() == 0) {
                    kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
                }

                ProfileSingleWorkload(profiler_prefix, workload, profiler_runner_ptr, FLAGS_FC_FORCE_PROFILER_CACHE);
            }
            else {
                LOG(INFO) << op_name_ << " from cache, not profile";
            }
        }
    }

    std::string GenOpFunction() override
    {
        return register_kernel_ptr_->GenKernelFunction(GetName(), context_ptr_->GetName(), GetAttrsMap());
    }

    std::unordered_map<std::string, std::any> GetAttrsMap()
    {
        std::unordered_map<std::string, std::any> op_attrs_map{{"dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"op_name", this->op_name_},
                                                               {"op_kind", this->op_kind_},
                                                               {"op_mode", this->op_mode_},
                                                               {"q_num_heads", this->q_num_heads_},
                                                               {"kv_num_heads", this->kv_num_heads_},
                                                               {"qk_head_dim", this->qk_head_dim_},
                                                               {"v_head_dim", this->v_head_dim_},
                                                               {"scale", this->scale_},
                                                               {"bias_enum", this->bias_enum_},
                                                               {"bias_rank_info", this->bias_rank_info_},
                                                               {"rotary_dim", this->rotary_dim_},
                                                               {"paged_block_size", this->paged_block_size_},
                                                               {"use_cache_batch_idx", this->use_cache_batch_idx_},
                                                               {"mask_enum", this->mask_enum_},
                                                               {"epilogue_op", this->epilogue_op_},
                                                               {"exec_path", this->exec_path_},
                                                               {"kernel_instance_map", this->kernel_instance_map_},
                                                               {"num_splits", this->num_splits_}};
        return op_attrs_map;
    }

    void Forward() override
    {
        static_cast<OpType*>(this)->ForwardImpl();
    }

    std::string op_name_;

    FmhaOperationMode op_mode_     = FmhaOperationMode::Batch;
    FmhaOperationKind op_kind_     = FmhaOperationKind::Fwd;
    TensorOperation   epilogue_op_ = TensorOperation::PassThrough;

    int64_t q_num_heads_;
    int64_t kv_num_heads_ = -1;
    int64_t qk_head_dim_;
    int64_t v_head_dim_ = -1;
    float   scale_      = 0.0f;

    BiasEnum bias_enum_ = BiasEnum::NO_BIAS;
    int64_t  bias_rank_info_;

    GenericAttentionMaskEnum mask_enum_   = GenericAttentionMaskEnum::NO_MASK;
    std::array<int64_t, 2>   window_size_ = {-1, -1};

    RopeEnum rope_enum_;
    int64_t  rotary_dim_ = -1;  // RoPE rotary dimension. rotary_dim <= 0 means not apply RoPE at all

    int64_t paged_block_size_ = -1;  // paged-kvcache block size. 0 means not use paged-kvcahe

    bool use_cache_batch_idx_ = false;  // whether to use index map to the kvcache

    int64_t num_splits_ = -1;  // splits for key/value. -1 to determine actual number by heuristic

    std::map<std::string, std::shared_ptr<RunningItem>> exec_path_;
    std::vector<std::string>                            exec_key_;

    std::shared_ptr<Kernel> register_kernel_ptr_;

    std::map<std::string, std::shared_ptr<void>> kernel_instance_map_;

    std::vector<Variable*> input_var_;
    std::vector<Variable*> output_var_;
};
}  // namespace flashck