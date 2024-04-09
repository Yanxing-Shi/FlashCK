#include "ater/core/profiler/emitters.h"

#include "ater/core/profiler/gemm_operation.h"
#include "ater/core/utils/log.h"

namespace ater {

// Get instance of Emitters
Emitters* Emitters::GetInstance()
{
    static Emitters emitter_instance = Emitters();
    return &emitter_instance;
}

// Inserts the kernel.
template<typename T>
void Emitters::Append(std::shared_ptr<T> kernel)
{
    std::string kernel_config_name = kernel->GetConfigName();
    kernel_names_vec_.emplace_back(kernel_config_name);
    VLOG(1) << "Appending kernel: " << kernel_config_name;

    // add the config name to the kernel config map
    kernel_instance_map_[kernel->operation_kind_][kernel->extra_kind_][kernel_config_name] = kernel;

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

std::map<OperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
Emitters::GetKernelInstanceMap() const
{
    return kernel_instance_map_;
}

template void Emitters::Append(std::shared_ptr<GemmOperation> kernel);

}  // namespace ater