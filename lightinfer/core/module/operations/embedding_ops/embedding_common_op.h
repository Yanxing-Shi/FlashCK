#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "lightinfer/core/graph/node.h"
#include "lightinfer/core/graph/shape.h"
#include "lightinfer/core/profiler/embedding_cache_entry.h"
#include "lightinfer/core/profiler/gpu_profiler_runner.h"
#include "lightinfer/core/profiler/target.h"

#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/file_utils.h"
#include "lightinfer/core/utils/flags.h"
#include "lightinfer/core/utils/log.h"

#include "lightinfer/core/module/kernels/embedding_kernels/embedding_add_add_layer_norm_kernel.h"
#include "lightinfer/core/module/kernels/embedding_kernels/embedding_kernel.h"

LI_DECLARE_bool(LI_FORCE_PROFILE);
LI_DECLARE_bool(LI_FORCE_PROFILER_CACHE);
LI_DECLARE_string(LI_HOME_PATH);

namespace lightinfer {

/*
embedding operation
*/

template<typename CppType, typename OpType>
class EmbeddingCommonOp: public Operation {
public:
    EmbeddingCommonOp(std::string op_name): Operation(op_name) {}
    virtual ~EmbeddingCommonOp() = default;

    Shape InferShape(Variable* a)
    {
        return a->GetShape().AppendDim({embedding_dims_});
    }

    std::vector<int64_t> InverseKeyFunc(const std::string& key)
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
                LI_THROW(Unavailable("norm input has empty dim values: {}", values[0]));
            }
        }

        return JoinToString(key_strs, " && ");
    }

    std::tuple<int64_t, int64_t> ComputeIndicesValues(Variable* x)
    {
        std::vector<DDim> dim_shape          = x->GetShape().ToVector();
        int64_t           max_indices_values = 1;
        int64_t           min_indices_values = 1;

        for (const auto& dim : dim_shape) {
            max_indices_values *= dim.GetUpperBound();
            min_indices_values *= dim.GetLowerBound();
        }

        return std::make_tuple(min_indices_values, max_indices_values);
    }

    void ExtractExecPath(const DynamicProfileStrategy& dynamic_profiling_strategy = DynamicProfileStrategy::MAX,
                         const int                     step_value                 = 1)
    {
        auto min_indices_values = std::get<0>(ComputeIndicesValues(GetParentNode(0)));
        auto max_indices_values = std::get<1>(ComputeIndicesValues(GetParentNode(0)));

        std::map<std::string, std::vector<int64_t>> shape_values_map = {
            {"num_indices", {min_indices_values, max_indices_values}},
        };

        if (dynamic_profiling_strategy == DynamicProfileStrategy::MAX) {
            std::map<std::string, std::vector<int64_t>> max_values = {
                {"num_indices", {max_indices_values}},
            };

            std::shared_ptr<ExecItem> exec_item_ptr =
                std::make_shared<ExecItem>(GenExecKey(max_values), GenExecKey(shape_values_map), "");

            exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
        }
        else if (dynamic_profiling_strategy == DynamicProfileStrategy::MIN) {
            std::map<std::string, std::vector<int64_t>> min_values = {
                {"num_indices", {min_indices_values}},
            };

            std::shared_ptr<ExecItem> exec_item_ptr =
                std::make_shared<ExecItem>(GenExecKey(min_values), GenExecKey(shape_values_map), "");

            exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
        }
        else if (dynamic_profiling_strategy == DynamicProfileStrategy::INTERATION) {
            std::map<std::string, std::vector<int64_t>> iter_values_map;
            for (int64_t indices = min_indices_values; indices <= max_indices_values; indices += step_value) {
                iter_values_map["num_indices"].push_back(indices);
            }

            // generate exec path
            std::vector<std::map<std::string, std::vector<int64_t>>> iter_values_vec;
            for (int64_t i = 0; i < max_indices_values; i++) {
                std::map<std::string, std::vector<int64_t>> iter_value{};
                for (const auto& [name, values] : iter_values_map) {
                    iter_value[name].push_back(values[i]);
                }
                iter_values_vec.push_back(iter_value);
            }

            for (const auto& iter_value : iter_values_vec) {
                std::shared_ptr<ExecItem> exec_item_ptr =
                    std::make_shared<ExecItem>(GenExecKey(iter_value), GenExecKey(iter_value), "");

                exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
            }
        }
        else {
            LI_THROW(Unimplemented("{}", "norm only supports MIN or MAX dynamic profiling"));
        }
    }

    void IfShouldBuildProfiler(const std::vector<std::string>& workloads)
    {
        for (const auto& workload : workloads) {
            auto query = static_cast<OpType*>(this)->GetEmbeddingQueryEntry(workload);

            auto cache_value = Target::Instance()->QueryProfileCache(GenOperationKind::Embedding, query);

            VLOG(1) << "cache_value: " << std::get<0>(cache_value) << "," << std::get<1>(cache_value);
            if (cache_value != std::make_tuple("null", -1) && !FLAGS_LI_FORCE_PROFILE) {
                std::string best_algo = std::get<0>(cache_value);
                LOG(INFO) << "Load profiling result for" << op_name_ << " from cache, algo: " << best_algo;

                exec_path_[workload]->algo_ = best_algo;
            }
            else {
                // cache miss - we will have to generate and build profilers
                LOG(INFO) << "No profiling result found for" << op_name_ << "in cache, will build profilers";
            }
        }
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenOpProfiler(const DynamicProfileStrategy& dynamic_profiling_strategy = DynamicProfileStrategy::MAX) override
    {
        KernelKey kernel_key(SourceType::CK, DataLayout::ALL_LAYOUT, CppTypeToDataType<CppType>::Type());
        register_kernel_ptr_ = KernelFactory::Instance().SelectKernel(op_name_, kernel_key);

        // init exec path
        ExtractExecPath(dynamic_profiling_strategy);

        exec_key_ = GetKeyList(exec_path_);

        if (!FLAGS_LI_FORCE_PROFILER_CACHE) {
            IfShouldBuildProfiler(exec_key_);
        }
        else {
            LOG(INFO) << "Forced to use cache, skip building profilers for " << op_name_;
            return {};
        }

        std::vector<std::vector<int64_t>> all_workloads;
        std::vector<EmbeddingProblem>     embedding_problems;
        all_workloads.reserve(exec_key_.size());
        embedding_problems.reserve(exec_key_.size());
        std::for_each(exec_key_.begin(), exec_key_.end(), [&](const std::string& key) {
            int64_t num_indices = InverseKeyFunc(key)[0];
            all_workloads.emplace_back(num_indices);

            EmbeddingProblem embedding_problem = static_cast<OpType*>(this)->GetEmbeddingProblem(num_indices);

            embedding_problems.emplace_back(embedding_problem);
        });

        std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> generated_profilers;

        for (int i = 0; i < exec_key_.size(); i++) {
            Target::Instance()->GenerateKernel(GenOperationKind::Embedding, embedding_problems[i]);

            // init kernel instance map
            kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
            if (kernel_instance_map_.size() == 0) {
                LI_THROW(Fatal("No Embedding op instances were generated for {}", op_name_));
            }
            if (exec_path_[exec_key_[i]]->algo_ == "") {
                // generate profiler
                generated_profilers = register_kernel_ptr_->GenKernelProfiler(context_ptr_->GetName(), GetAttrsMap());
            }
            else {
                LOG(INFO) << "op_name: " << op_name_ << ", " << "workload: " << exec_key_[i]
                          << " from cache, not profile";
            }
        }

        return generated_profilers;
    }

    void ProfileSingleWorkload(const std::string&                        profiler_prefix,
                               const std::string&                        workload,
                               const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                               bool                                      force_cache)
    {
        std::vector<std::string> kernel_instance_map_key = GetKeyList(kernel_instance_map_);

        EmbeddingQueryEntry query = static_cast<OpType*>(this)->GetEmbeddingQueryEntry(workload);

        auto cache_value = Target::Instance()->QueryProfileCache(GenOperationKind::Embedding, query);

        if (cache_value == std::make_tuple("null", -1) && force_cache) {
            LOG(ERROR)
                << "force_cache is enabled but we could not find the following cache available on device. op_name:"
                << op_name_ << "workload: " << workload;
        }

        for (const auto& kernel_config_name : kernel_instance_map_key) {
            auto GenCallback = [&]() {
                auto process_result_callback =
                    [&](const std::vector<ProfileResult>&           result,
                        const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr) {
                        postprocessing_delegate_ptr->AddInstance(
                            result, GenOperationKind::Embedding, GetAttrsMap(), kernel_config_name, workload, split_k_);
                    };
                return process_result_callback;
            };

            int64_t                  num_indices = InverseKeyFunc(workload)[0];
            std::vector<std::string> command =
                static_cast<OpType*>(this)->GenOpProfileCmd(profiler_prefix, kernel_config_name, num_indices);

            LOG(INFO) << "profile command: " << JoinToString(command);

            profiler_runner_ptr->Push(command, GenCallback());
        }
    }

    void Profile(const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                 const std::string&                        folder_name = "kernel_profile") override
    {
        std::filesystem::path profiler_prefix =
            std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / context_ptr_->GetName() / "profiler" / op_name_;

        for (const auto& workload : exec_key_) {
            if (exec_path_[workload]->algo_ == "") {
                if (kernel_instance_map_.size() == 0) {
                    kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
                }

                ProfileSingleWorkload(profiler_prefix, workload, profiler_runner_ptr, FLAGS_LI_FORCE_PROFILER_CACHE);
            }
            else {
                LOG(INFO) << "op: " << op_name_ << ", worload: " << workload << " from cache, not profile";
            }
        }
    }

    std::string GenOpFunction() override
    {
        return register_kernel_ptr_->GenKernelFunction(GetName(), context_ptr_->GetName(), GetAttrsMap());
    }

    void Forward() override
    {
        static_cast<OpType*>(this)->ForwardImpl();
    }

    std::unordered_map<std::string, std::any> GetAttrsMap()
    {
        std::unordered_map<std::string, std::any> op_attrs_map{{"num_embeddings", num_embeddings_},
                                                               {"vocab_size", vocab_size_},
                                                               {"type_vocab_size", type_vocab_size_},
                                                               {"max_position_embeddings", max_position_embeddings_},
                                                               {"embedding_dims", embedding_dims_},
                                                               {"epilogue_op", epilogue_op_},
                                                               {"emb_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"index_dtype", DataType::INT64},
                                                               {"weight_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"gamma_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"beta_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"acc_dtype", DataType::FLOAT32},
                                                               {"y_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"op_name", op_name_},
                                                               {"exec_path", exec_path_},
                                                               {"kernel_instance_map", kernel_instance_map_},
                                                               {"epsilon", epsilon_}};
        return op_attrs_map;
    }

    int64_t num_embeddings_          = -1;
    int64_t vocab_size_              = -1;
    int64_t type_vocab_size_         = -1;
    int64_t max_position_embeddings_ = -1;

    int64_t embedding_dims_ = -1;
    int64_t split_k_        = 1;

    float epsilon_ = 1e-12;

    std::string            op_name_;
    EmbeddingOperationKind op_kind_     = EmbeddingOperationKind::SparseEmbedding;
    TensorOperation        epilogue_op_ = TensorOperation::AddAddLayerNorm;

    bool is_should_build_profiler_ = true;

    std::map<std::string, std::shared_ptr<ExecItem>> exec_path_;
    std::vector<std::string>                         exec_key_;

    std::shared_ptr<Kernel> register_kernel_ptr_;

    std::map<std::string, std::shared_ptr<void>> kernel_instance_map_;

    std::vector<Variable*> input_var_;
    std::vector<Variable*> output_var_;
};

}  // namespace lightinfer