#pragma once

#include <any>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "lightinfer/core/graph/shape.h"

#include "lightinfer/core/profiler/base.h"
#include "lightinfer/core/profiler/library.h"
#include "lightinfer/core/profiler/target.h"

#include "lightinfer/core/utils/enforce.h"

#include "lightinfer/core/module/kernels/kernel_args.h"
#include "lightinfer/core/module/kernels/kernel_call_def.h"

namespace lightinfer {

class Kernel {
public:
    using OperationKind = std::variant<EmbeddingOperationKind, GemmOperationKind, NormOperationKind, FmhaOperationKind>;
    using KernelArgs    = std::variant<EmbeddingKernelArgs,
                                       GemmKernelArgs,
                                       NormKernelArgs,
                                       FmhaFwdKernelArgs,
                                       FmhaFwdAppendKVKernelArgs,
                                       FmhaFwdSplitKVKernelArgs,
                                       FmhaFwdSplitKVCombineKernelArgs>;

    Kernel()          = default;
    virtual ~Kernel() = default;

    // step.1 Init Kernel
    virtual std::map<std::string, std::shared_ptr<void>> Init(const OperationKind&   op_kind,
                                                              const TensorOperation& extra_kind)
    {
        LI_THROW(Unimplemented("Kernel base init is not implemented."));
    }

    // step.2 GenKernelProfiler
    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenKernelProfiler(const std::string&                               model_name,
                      const std::unordered_map<std::string, std::any>& kernel_func_map,
                      const std::string&                               folder_name = "kernel_profile")
    {
        LI_THROW(Unimplemented("Kernel base GenKernelProfiler is not implemented."));
    }

    // step.3 GenKernelFunction
    virtual std::string GenKernelFunction(const std::string&                               func_name,
                                          const std::string&                               model_name,
                                          const std::unordered_map<std::string, std::any>& kernel_func_map)
    {
        LI_THROW(Unimplemented("Kernel base GenKernelFunction is not implemented."));
    }

    // step.4 KernelLauncher
    virtual void KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
    {
        LI_THROW(Unimplemented("Kernel base KernelLauncher is not implemented."));
    }
};

}  // namespace lightinfer