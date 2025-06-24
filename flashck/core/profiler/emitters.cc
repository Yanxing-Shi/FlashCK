#include "flashck/core/profiler/emitters.h"

#include "flashck/core/utils/log.h"

namespace flashck {

// Get instance of Emitters
Emitters* Emitters::GetInstance()
{
    static Emitters emitter_instance = Emitters();
    return &emitter_instance;
}

// Inserts the kernel.
template<>
void Emitters::Append(std::shared_ptr<GemmOperation> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    std::string kernel_config      = kernel->Emit();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;
    // VLOG(1) << "Appending kernel config: " << kernel_config;

    // add the config name to the kernel config map
    gemm_kernel_instance_map_[kernel->operation_kind_][kernel->epilogue_op_][kernel_config_name] = kernel;

    kernel_count_map_[kernel_config_name] += 1;

    // increment the total kernel count
    total_kernel_count_ += 1;
}

template<>
void Emitters::Append(std::shared_ptr<NormOperation> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;

    // add the config name to the kernel config map
    norm_kernel_instance_map_[kernel->operation_kind_][kernel->epilogue_op_][kernel_config_name] = kernel;

    kernel_count_map_[kernel_config_name] += 1;

    // increment the total kernel count
    total_kernel_count_ += 1;
}

template<>
void Emitters::Append(std::shared_ptr<EmbeddingOperation> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;

    // add the config name to the kernel config map
    embedding_kernel_instance_map_[kernel->operation_kind_][kernel->epilogue_op_][kernel_config_name] = kernel;

    kernel_count_map_[kernel_config_name] += 1;

    // increment the total kernel count
    total_kernel_count_ += 1;
}

template<>
void Emitters::Append(std::shared_ptr<FmhaFwdOperation> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;

    // add the config name to the kernel config map
    fmha_kernel_instance_map_[kernel->operation_kind_][kernel->epilogue_op_][kernel_config_name] = kernel;

    kernel_count_map_[kernel_config_name] += 1;

    // increment the total kernel count
    total_kernel_count_ += 1;
}

template<>
void Emitters::Append(std::shared_ptr<FmhaFwdSplitKVOperation> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;

    // add the config name to the kernel config map
    fmha_kernel_instance_map_[kernel->operation_kind_][kernel->epilogue_op_][kernel_config_name] = kernel;

    kernel_count_map_[kernel_config_name] += 1;

    // increment the total kernel count
    total_kernel_count_ += 1;
}

template<>
void Emitters::Append(std::shared_ptr<FmhaFwdSplitKVCombineOperation> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;

    // add the config name to the kernel config map
    fmha_kernel_instance_map_[kernel->operation_kind_][kernel->epilogue_op_][kernel_config_name] = kernel;

    kernel_count_map_[kernel_config_name] += 1;

    // increment the total kernel count
    total_kernel_count_ += 1;
}

template<>
void Emitters::Append(std::shared_ptr<FmhaFwdAppendKVOperation> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;

    // add the config name to the kernel config map
    fmha_kernel_instance_map_[kernel->operation_kind_][kernel->epilogue_op_][kernel_config_name] = kernel;

    kernel_count_map_[kernel_config_name] += 1;

    // increment the total kernel count
    total_kernel_count_ += 1;
}

// Get the config of kernel
std::vector<std::string> Emitters::GetConfigNameVec() const
{
    return kernel_names_vec_;
}

// Get the number of generate kernel
int Emitters::GetTotalKernelCount() const
{
    return total_kernel_count_;
}

// Get the map of kernel count
std::map<std::string, int> Emitters::GetKernelCountMap() const
{
    return kernel_count_map_;
}

std::map<GemmOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
Emitters::GetGemmKernelInstanceMap() const
{
    return gemm_kernel_instance_map_;
}

std::map<NormOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
Emitters::GetNormKernelInstanceMap() const
{
    return norm_kernel_instance_map_;
}

std::map<EmbeddingOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
Emitters::GetEmbeddingKernelInstanceMap() const
{
    return embedding_kernel_instance_map_;
}

std::map<FmhaOperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
Emitters::GetFmhaFwdKernelInstanceMap() const
{
    return fmha_kernel_instance_map_;
}

}  // namespace flashck