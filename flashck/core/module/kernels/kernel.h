#pragma once

#include <any>

#include "flashck/core/graph/shape.h"
#include "flashck/core/utils/common.h"

#include "flashck/core/profiling/library.h"

#include "flashck/core/module/kernels/kernel_args.h"
#include "flashck/core/module/kernels/kernel_call_def.h"

namespace flashck {

class Kernel {
public:
    using kernelKind = std::variant<NormKind>;
    using KernelArgs = std::variant<NormKernelArgs>;

    Kernel()          = default;
    virtual ~Kernel() = default;

    // step.2 GenKernelProfiler
    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenKernelProfiler(const std::string&                               model_name,
                      const std::unordered_map<std::string, std::any>& kernel_func_map,
                      const std::string&                               folder_name = "kernel_profile")
    {
        FC_THROW(Unimplemented("Kernel base GenKernelProfiler is not implemented."));
    }

    // step.3 GenKernelFunction
    virtual std::string GenKernelFunction(const std::string&                               func_name,
                                          const std::string&                               model_name,
                                          const std::unordered_map<std::string, std::any>& kernel_func_map)
    {
        FC_THROW(Unimplemented("Kernel base GenKernelFunction is not implemented."));
    }

    // step.4 KernelLauncher
    virtual void KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
    {
        FC_THROW(Unimplemented("Kernel base KernelLauncher is not implemented."));
    }
};

}  // namespace flashck