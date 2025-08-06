#pragma once

#include "core/module/kernels/kernel_registry.h"
#include "core/module/kernels/kernel.h"

#include "core/module/kernels/attention_kernels/fmha_common_kernel.h"


namespace flashck {
class FmhaFwdAppendKVKernel: public FmhaCommonKernel {
public:
    FmhaFwdAppendKVKernel()  = default;
    ~FmhaFwdAppendKVKernel() = default;

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const std::string&    model_name,
                     const instance_map_t& instance_map,
                     const std::string&    folder_name = "kernel_profile") override;

    std::string CodeGenForRunning(const std::string&                        func_name,
                                  const std::string&                        model_name,
                                  const std::map<std::string, RunningItem>& running_infos,
                                  const instance_map_t&                     instance_map,
                                  const std::string&                        folder_name = "kernel_profile") override;

    void KernelLauncher(const std::string& kernel_func_name, const KernelArgs_t& args) override;
};

}  // namespace flashck

FC_REGISTER_KERNEL(TILE, fmha_fwd_appendkv, flashck::FmhaFwdAppendKVKernel, ALL_LAYOUT, FP16, BF16);