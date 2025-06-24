#include "flashck/core/module/kernels/gemm_kernels/bmm_epilogue_kernels/bmm_common_kernel.h"

#include "flashck/core/utils/dylib_utils.h"
#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/log.h"

LI_DECLARE_string(LI_HOME_PATH);

namespace flashck {

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
BmmCommonKernel::GenBmmCommonKernelProfiler(const std::string&                               model_name,
                                            const std::unordered_map<std::string, std::any>& kernel_func_map,
                                            const std::string&                               arg_parse,
                                            const std::string&                               gemm_flag,
                                            const std::string&                               extra_code,
                                            const std::string&                               extra_shape_template,
                                            const std::string&                               problem_args_template,
                                            const std::string&                               extra_header_template,
                                            const std::string&                               tensor_decl_template)
{
    return GenGemmCommonKernelProfiler(model_name,
                                       kernel_func_map,
                                       arg_parse,
                                       gemm_flag,
                                       extra_code,
                                       3,
                                       extra_shape_template,
                                       problem_args_template,
                                       extra_header_template,
                                       tensor_decl_template);
}

std::string BmmCommonKernel::GenBmmKernelFunction(const std::string&                               func_name,
                                                  const std::unordered_map<std::string, std::any>& kernel_func_map,
                                                  const std::string&                               gemm_flag,
                                                  const std::string&                               extra_code,
                                                  const std::string&                               extra_shape_template,
                                                  const std::string& problem_args_template,
                                                  const std::string& extra_header_template)
{
    return GenGemmCommonKernelFunction(func_name,
                                       kernel_func_map,
                                       gemm_flag,
                                       extra_code,
                                       3,
                                       extra_shape_template,
                                       problem_args_template,
                                       extra_header_template);
}

}  // namespace flashck