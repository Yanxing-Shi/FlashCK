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
    GenGemmCommonKernelProfiler(const std::string&                               model_name,
                                const std::unordered_map<std::string, std::any>& kernel_func_map,
                                const std::string&                               arg_parse,
                                const std::string&                               gemm_flag  = "",
                                const std::string&                               extra_code = "",
                                const int                                        ndims      = 2,
                                const std::string& extra_shape_template                     = g_extra_shape_source,
                                const std::string& problem_args_template                    = g_problem_args_source,
                                const std::string& extra_header_template                    = g_extra_header_source,
                                const std::string& tensor_decl_template                     = g_tensor_decl_source,
                                const std::string& inverse_shape                            = "",
                                const std::string& input_addr_calculator                    = "",
                                const std::string& output_addr_calculator                   = "",
                                const std::string& folder_name                              = "kernel_profile");

    std::string GenGemmCommonKernelFunction(const std::string&                               func_name,
                                            const std::unordered_map<std::string, std::any>& kernel_func_map,
                                            const std::string&                               gemm_flag  = "",
                                            const std::string&                               extra_code = "",
                                            const int                                        ndims      = 2,
                                            const std::string& extra_shape_template  = g_extra_shape_source,
                                            const std::string& problem_args_template = g_problem_args_source,
                                            const std::string& extra_header_template = g_extra_header_source,
                                            const std::string& inverse_shape         = "",
                                            const std::string& exec_cond_template    = g_exec_cond_source);
};
}  // namespace lightinfer