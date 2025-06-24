#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "flashck/core/profiler/embedding_operation.h"
#include "flashck/core/profiler/fmha_fwd_appendkv_operation.h"
#include "flashck/core/profiler/fmha_fwd_operation.h"
#include "flashck/core/profiler/fmha_fwd_splitkv_combine_operation.h"
#include "flashck/core/profiler/fmha_fwd_splitkv_operation.h"
#include "flashck/core/profiler/gemm_operation.h"
#include "flashck/core/profiler/library.h"
#include "flashck/core/profiler/norm_operation.h"

#include "flashck/core/utils/string_utils.h"

namespace flashck {

class Emitters {
public:
    Emitters() = default;

    // Get instance of Emitters
    static Emitters* GetInstance();

    // Inserts the kernel
    template<typename T>
    void Append(std::shared_ptr<T> kernel);

    // Get the config of kernel
    std::vector<std::string> GetConfigNameVec() const;

    // Get the number of generate kernel
    int GetTotalKernelCount() const;

    // Get the map of kernel count
    std::map<std::string, int> GetKernelCountMap() const;

    // Get the kernel instance map
    std::map<GemmOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
    GetGemmKernelInstanceMap() const;

    std::map<EmbeddingOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
    GetEmbeddingKernelInstanceMap() const;

    std::map<NormOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
    GetNormKernelInstanceMap() const;

    std::map<FmhaOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
    GetFmhaFwdKernelInstanceMap() const;

private:
    std::map<GemmOperationKind, std::map<std::string, std::shared_ptr<void>>>      gemm_kernel_instance_map_;
    std::map<EmbeddingOperationKind, std::map<std::string, std::shared_ptr<void>>> embedding_kernel_instance_map_;
    std::map<NormOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        norm_kernel_instance_map_;
    std::map<FmhaOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        fmha_kernel_instance_map_;

    int                        total_kernel_count_;
    std::map<std::string, int> kernel_count_map_;
    std::vector<std::string>   kernel_names_vec_;
};

}  // namespace flashck