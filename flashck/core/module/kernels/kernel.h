#pragma once

#include "flashck/core/utils/common.h"

#include "flashck/core/module/kernels/kernel_args.h"
#include "flashck/core/module/kernels/kernel_call_def.h"

namespace flashck {

struct TuningTpl {
    std::string dtype_config_tpl_;
    std::string dtype_decl_tpl_;
    std::string func_signature_tpl_;
    std::string make_args_tpl_;
    std::string tensor_decl_tpl_;
    std::string func_call_tpl_;
};

struct RunningTpl {
    std::string dtype_config_tpl_;
    std::string dtype_decl_tpl_;
    std::string func_signature_tpl_;
    std::string make_args_tpl_;
};

// This structure is used to store the execution items for profiling
struct RunningItem {
    std::string profiling_key_;

    std::string running_cond_;
    std::string instance_name_;
    PerfResult  perf_result_;
};

class Kernel {
public:
    using KernelArgs = std::variant<NormKernelArgs>;

    Kernel()          = default;
    virtual ~Kernel() = default;

    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const std::string&                                  model_name,
                     const std::string&                                  kind_name,
                     const std::map<std::string, std::unique_ptr<void>>& instance_map,
                     const std::string&                                  folder_name = "kernel_profile")
    {
        FC_THROW(Unimplemented("Kernel base CodeGenForTuning is not implemented."));
    }

    virtual std::string CodeGenForRunning(const std::string&                                  func_name,
                                          const std::string&                                  model_name,
                                          const std::vector<RunningItem>&                     running_items,
                                          const std::map<std::string, std::unique_ptr<void>>& kernel_instance_map,
                                          const std::string& folder_name = "kernel_profile")
    {
        FC_THROW(Unimplemented("Kernel base CodeGenForRunning is not implemented."));
    }

    virtual void KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
    {
        FC_THROW(Unimplemented("Kernel base KernelLauncher is not implemented."));
    }
};

}  // namespace flashck