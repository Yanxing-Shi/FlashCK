#pragma once

#include <any>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "flashck/core/graph/shape.h"

#include "flashck/core/profiling/base.h"
#include "flashck/core/profiling/library.h"
#include "flashck/core/profiling/target.h"

#include "flashck/core/utils/enforce.h"

#include "flashck/core/module/kernels/kernel_args.h"
#include "flashck/core/module/kernels/kernel_call_def.h"

namespace flashck {

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
        FC_THROW(Unimplemented("Kernel base init is not implemented."));
    }

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