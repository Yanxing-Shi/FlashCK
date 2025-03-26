
#include "lightinfer/core/profiler/gpu_profiler_runner.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>

#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/flags.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/printf.h"
#include "lightinfer/core/utils/string_utils.h"

#include "lightinfer/core/graph/shape.h"
#include "lightinfer/core/profiler/base.h"
#include "lightinfer/core/profiler/target.h"

namespace lightinfer {

// As a profiler executable completes, collect the result
void ProfilerPostprocess::AddInstance(const std::vector<ProfileResult>&                result,
                                      const GenOperationKind&                          op_kind,
                                      const std::unordered_map<std::string, std::any>& op_attrs_map,
                                      const std::string&                               kernel_name,
                                      const std::string&                               exec_key,
                                      const int64_t                                    split_k)
{
    instances_.emplace_back(std::make_tuple(result, op_kind, op_attrs_map, kernel_name, exec_key, split_k));
}

struct ProfileResultGroupByKey {
    typedef std::tuple<GenOperationKind, std::string, std::string> value_type;
    value_type operator()(const std::tuple<std::vector<ProfileResult>,
                                           GenOperationKind,
                                           std::unordered_map<std::string, std::any>,
                                           std::string,
                                           std::string,
                                           int64_t>& instance) const
    {
        return std::make_tuple(std::get<1>(instance),
                               std::any_cast<std::string>(std::get<2>(instance).at("op_name")),  // unique op name
                               std::get<4>(instance));  // profiler key (gemm shape)
    }
};

void ProfilerPostprocess::PostProcessResults()
{
    auto instance_group_by_key = GroupByFunc(instances_.begin(), instances_.end(), ProfileResultGroupByKey());

    for (auto group : instance_group_by_key) {
        // select minimal duration algo
        auto min_runtime_results = *std::min_element(
            group.range.begin(),
            group.range.end(),
            [](const std::tuple<std::vector<ProfileResult>,
                                GenOperationKind,
                                std::unordered_map<std::string, std::any>,
                                std::string,
                                std::string,
                                int64_t>& a,
               const std::tuple<std::vector<ProfileResult>,
                                GenOperationKind,
                                std::unordered_map<std::string, std::any>,
                                std::string,
                                std::string,
                                int64_t>& b) { return std::get<0>(a)[0].duration < std::get<0>(b)[0].duration; });
        // only gemm profile result
        ProfileResult best_profile_result = std::get<0>(min_runtime_results)[0];

        GenOperationKind                          op_kind           = std::get<1>(min_runtime_results);
        std::unordered_map<std::string, std::any> op_attrs_map      = std::get<2>(min_runtime_results);
        std::string                               profiler_filename = std::get<3>(min_runtime_results);
        std::string                               exec_key          = std::get<4>(min_runtime_results);
        int64_t                                   split_k           = std::get<5>(min_runtime_results);

        for (auto profile_instance : group.range) {
            auto op_attrs_map = std::get<2>(profile_instance);
            std::any_cast<std::map<std::string, std::shared_ptr<ExecItem>>>(op_attrs_map["exec_path"])[exec_key]
                ->algo_ = best_profile_result.kernel_config;
            std::any_cast<std::map<std::string, std::shared_ptr<ExecItem>>>(op_attrs_map["exec_path"])[exec_key]
                ->split_k_ = split_k;
        }

        LOG(INFO) << "Profiler " << profiler_filename << " selected kernel " << best_profile_result.kernel_config
                  << " with duration " << best_profile_result.duration << " ms, " << "split_k: " << split_k;

        if (op_kind == GenOperationKind::Gemm) {
            std::string exec_entry_sha1 = SHA1ToHexString(exec_key);
            auto        cache_record    = GemmRecordEntry(
                exec_key,
                exec_entry_sha1,
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["a_dtype"])),
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["b_dtype"])),
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["c_dtype"])),
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["acc_dtype"])),
                DataLayoutToString(std::any_cast<DataLayout>(op_attrs_map["layout"])),
                std::any_cast<std::string>(op_attrs_map["op_name"]),
                g_short_tensor_operation_names_map.at(std::any_cast<TensorOperation>(op_attrs_map["epilogue_op"])),
                std::any_cast<Shape>(op_attrs_map["permute_shape"]).ToString(),
                Target::Instance()->GetTargetDeviceName(),
                best_profile_result.kernel_config,
                split_k);
            try {
                Target::Instance()->InsertProfileCache(op_kind, cache_record);
            }
            catch (const std::exception& e) {
                LI_THROW(Unavailable("Failed to update record cache:{}", e.what()));
            }
        }
        else if (op_kind == GenOperationKind::Fmha) {
            std::string exec_entry_sha1 = SHA1ToHexString(exec_key);
            auto        cache_record    = FmhaRecordEntry(
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["dtype"])),
                g_generic_attention_mask_short_names_map.at(
                    std::any_cast<GenericAttentionMaskEnum>(op_attrs_map["mask_enum"])),
                g_bias_enum_names_map.at(std::any_cast<BiasEnum>(op_attrs_map["bias_enum"])),
                g_fmha_operation_mode_name_map.at(std::any_cast<FmhaOperationMode>(op_attrs_map["op_mode"])),
                std::any_cast<int64_t>(op_attrs_map["rotary_dim"]),
                std::any_cast<int64_t>(op_attrs_map["paged_block_size"]),
                std::any_cast<bool>(op_attrs_map["use_cache_batch_idx"]),
                g_fmha_kind_names_map.at(std::any_cast<FmhaOperationKind>(op_attrs_map["op_kind"])),
                Target::Instance()->GetTargetDeviceName(),
                g_short_tensor_operation_names_map.at(std::any_cast<TensorOperation>(op_attrs_map["epilogue_op"])),
                exec_key,
                exec_entry_sha1,
                split_k,
                best_profile_result.kernel_config);
            try {
                Target::Instance()->InsertProfileCache(op_kind, cache_record);
            }
            catch (const std::exception& e) {
                LI_THROW(Unavailable("Failed to update record cache:{}", e.what()));
            }
        }

        else if (op_kind == GenOperationKind::Embedding) {
            auto embebdding_epilogue_op = std::any_cast<TensorOperation>(op_attrs_map["epilogue_op"]);
            if (embebdding_epilogue_op == TensorOperation::PassThrough) {
                std::string exec_entry_sha1 = SHA1ToHexString(exec_key);
                auto        cache_record    = EmbeddingRecordEntry(
                    exec_key,
                    exec_entry_sha1,
                    std::any_cast<int64_t>(op_attrs_map["num_embeddings"]),
                    std::any_cast<int64_t>(op_attrs_map["embedding_dims"]),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["emb_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["index_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["gamma_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["beta_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["acc_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["y_dtype"])),
                    std::any_cast<std::string>(op_attrs_map["op_name"]),
                    g_short_tensor_operation_names_map.at(std::any_cast<TensorOperation>(op_attrs_map["epilogue_op"])),
                    Target::Instance()->GetTargetDeviceName(),
                    best_profile_result.kernel_config);
                try {
                    Target::Instance()->InsertProfileCache(op_kind, cache_record);
                }
                catch (const std::exception& e) {
                    LI_THROW(Unavailable("Failed to update record cache:{}", e.what()));
                }
            }
            else if (embebdding_epilogue_op == TensorOperation::AddAddLayerNorm) {
                std::string exec_entry_sha1 = SHA1ToHexString(exec_key);
                auto        cache_record    = EmbeddingRecordEntry(
                    exec_key,
                    exec_entry_sha1,
                    std::any_cast<int64_t>(op_attrs_map["vocab_size"]),
                    std::any_cast<int64_t>(op_attrs_map["type_vocab_size"]),
                    std::any_cast<int64_t>(op_attrs_map["max_position_embeddings"]),
                    std::any_cast<int64_t>(op_attrs_map["embedding_dims"]),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["emb_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["index_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["gamma_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["beta_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["acc_dtype"])),
                    DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["y_dtype"])),
                    std::any_cast<std::string>(op_attrs_map["op_name"]),
                    g_short_tensor_operation_names_map.at(std::any_cast<TensorOperation>(op_attrs_map["epilogue_op"])),
                    Target::Instance()->GetTargetDeviceName(),
                    best_profile_result.kernel_config);
                try {
                    Target::Instance()->InsertProfileCache(op_kind, cache_record);
                }
                catch (const std::exception& e) {
                    LI_THROW(Unavailable("Failed to update record cache:{}", e.what()));
                }
            }
        }
        else if (op_kind == GenOperationKind::Norm) {
            std::string exec_entry_sha1 = SHA1ToHexString(exec_key);
            auto        cache_record    = NormRecordEntry(
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["x_dtype"])),
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["y_dtype"])),
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["smooth_scale_dtype"])),
                DataTypeToShortString(std::any_cast<DataType>(op_attrs_map["y_scale_dtype"])),
                g_norm_operation_kind_names_map.at(std::any_cast<NormOperationKind>(op_attrs_map["op_kind"])),
                Target::Instance()->GetTargetDeviceName(),
                g_short_tensor_operation_names_map.at(std::any_cast<TensorOperation>(op_attrs_map["epilogue_op"])),
                exec_key,
                exec_entry_sha1,
                g_fused_add_enum_str_map.at(std::any_cast<FusedAddEnum>(op_attrs_map["fused_add"])),
                g_fused_quant_enum_str_map.at(std::any_cast<FusedQuantEnum>(op_attrs_map["fused_quant"])),
                best_profile_result.kernel_config);
            try {
                Target::Instance()->InsertProfileCache(op_kind, cache_record);
            }
            catch (const std::exception& e) {
                LI_THROW(Unavailable("Failed to update record cache:{}", e.what()));
            }
        }
    }
}

GPUProfilerRunner::GPUProfilerRunner(const int                                   num_gpu,
                                     const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr,
                                     const int                                   timeout):
    num_gpu_(num_gpu), timeout_(timeout), postprocessing_ptr_(postprocessing_delegate_ptr)
{
}

void GPUProfilerRunner::Push(
    const std::vector<std::string>& cmds,
    const std::function<void(const std::vector<ProfileResult>&, const std::shared_ptr<ProfilerPostprocess>&)>&
        process_result_callback)
{
    auto run_task = [this, &cmds, &process_result_callback]() {
        LOG(INFO) << "running profiler " << JoinToString(cmds);
        int attempts = 0;
        try {
            subprocess::Popen popen(cmds, subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});

            std::string stdout_str = popen.communicate().first.buf.data();
            std::string stderr_str = popen.communicate().second.buf.data();
            if (stdout_str.empty() && stderr_str.empty()) {
                LOG(ERROR) << "Profiler failed to run, stdout or stderr is empty";
            }
            else {
                LOG(INFO) << "Profiler stdout: " << stdout_str;
                // collect profiler results for postprocessing
                std::vector<ProfileResult> profile_result;
                bool                       is_error = false;
                std::tie(profile_result, is_error)  = ExtractProfileResult(stdout_str);
                if (is_error) {
                    LOG(ERROR) << "Profiler failure!\nProfiler stdout: " << stdout_str
                               << "\nProfiler stderr: " << stderr_str;
                }
                process_result_callback(profile_result, postprocessing_ptr_);
            }
        }
        catch (std::exception& e) {
            attempts += 1;
            if (attempts >= g_profiler_run_max_attempts) {
                LI_THROW(ExecutionTimeout(
                    "[{} / {}] Failed to run profiler {} due to exception: {}. Will retry in {} seconds.",
                    attempts,
                    g_profiler_run_max_attempts,
                    g_profiler_run_retry_delay_seconds,
                    JoinToString(cmds),
                    e.what()));
            }
            std::this_thread::sleep_for(std::chrono::seconds(g_profiler_run_retry_delay_seconds));
        }
    };

    run_task();
}

void GPUProfilerRunner::Join()
{
    postprocessing_ptr_->PostProcessResults();
}

}  // namespace lightinfer