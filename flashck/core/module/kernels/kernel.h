#pragma once

#include "flashck/core/module/kernels/kernel_args.h"
#include "flashck/core/module/kernels/kernel_call_def.h"
#include "flashck/core/utils/common.h"

namespace flashck {

// Template configuration for kernel tuning
struct TuningTpl {
    std::string dtype_config_tpl_;
    std::string dtype_decl_tpl_;
    std::string func_signature_tpl_;
    std::string make_args_tpl_;
    std::string tensor_decl_tpl_;
    std::string func_call_tpl_;

    TuningTpl() = default;
    TuningTpl(const std::vector<std::string>& templates)
    {
        if (templates.size() >= 6) {
            dtype_config_tpl_   = templates[0];
            dtype_decl_tpl_     = templates[1];
            func_signature_tpl_ = templates[2];
            make_args_tpl_      = templates[3];
            tensor_decl_tpl_    = templates[4];
            func_call_tpl_      = templates[5];
        }
    }
};

// Template configuration for kernel running
struct RunningTpl {
    std::string dtype_config_tpl_;
    std::string dtype_decl_tpl_;
    std::string func_signature_tpl_;
    std::string make_args_tpl_;

    RunningTpl() = default;
    RunningTpl(const std::vector<std::string>& templates)
    {
        if (templates.size() >= 4) {
            dtype_config_tpl_   = templates[0];
            dtype_decl_tpl_     = templates[1];
            func_signature_tpl_ = templates[2];
            make_args_tpl_      = templates[3];
        }
    }
};

// Execution item for profiling
struct RunningItem {
    std::string profiling_workload_;
    std::string running_cond_;
    std::string instance_name_;
    PerfResult  perf_result_;
};

// Base kernel class
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
                                          const std::map<std::string, RunningItem>&           running_infos,
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