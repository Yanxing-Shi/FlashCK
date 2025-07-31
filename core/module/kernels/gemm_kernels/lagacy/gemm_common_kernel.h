#include "lightinfer/core/module/kernels/kernel.h"


namespace flashck {

class LegacyGemmCommonKernel: public Kernel {
public:
    LegacyGemmCommonKernel()          = default;
    virtual ~LegacyGemmCommonKernel() = default;

    std::string GenDimCalculator(const std::shared_ptr<DimInfo>& dim_info, bool is_ptr);

    std::string GenShapeEvalCode(const std::string&                                                  dtype,
                                 const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
                                 bool                                                                is_ptr);

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenGemmCommonKernelProfiler(const std::string&    model_name,
                           const std::string&    kind_name,
                           const instance_map_t& instance_map,
                           const LegacyGemmTuningTpl&      tuning_tpl,
                           const std::string&    folder_name = "kernel_profile");

    std::string GenGemmCommonKernelFunction(const std::string&                        func_name,
                                            const std::string&                        model_name,
                                            const std::map<std::string, RunningItem>& running_infos,
                                            const instance_map_t&                     instance_map,
                                            const LegacyGemmRunningTpl&                     running_tpl,
                                            const std::string&                        folder_name = "kernel_profile");
};
}  // namespace flashck