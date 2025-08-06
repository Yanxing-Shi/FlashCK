
#pragma once

#include "lightinfer/core/module/kernels/kernel.h"
#include "lightinfer/core/module/kernels/kernel_registry.h"

#include "lightinfer/core/module/kernels/attention_kernels/fmha_common_jinja.h"


namespace flashck {

class FmhaCommonKernel: public Kernel {
public:

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CommonCodeGenForTuning(const std::string&    model_name,
                           const std::string&    kind_name,
                           const instance_map_t& instance_map,
                           const FmhaTuningTpl&      tuning_tpl,
                           const std::string&    folder_name = "kernel_profile");

    std::string CommonCodeGenForRunning(const std::string&                        func_name,
                                        const std::string&                        model_name,
                                        const std::map<std::string, RunningItem>& running_infos,
                                        const instance_map_t&                     instance_map,
                                        const FmhaRunningTpl&                         running_tpl,
                                        const std::string&                        folder_name = "kernel_profile");
};
}  // namespace flashck