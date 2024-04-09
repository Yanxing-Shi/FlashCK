#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ater/core/profiler/library.h"

#include "ater/core/utils/string_utils.h"

namespace ater {

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
    std::map<OperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
    GetKernelInstanceMap() const;

private:
    std::map<OperationKind, std::map<TensorOperation, std::map<std::string, std::shared_ptr<void>>>>
        kernel_instance_map_;

    int                        total_kernel_count_;
    std::map<std::string, int> kernel_count_map_;
    std::vector<std::string>   kernel_names_vec_;
};

}  // namespace ater