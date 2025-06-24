#include "flashck/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_add_kernel.h"

#include "flashck/core/module/kernels/gemm_kernels/layout.h"

/*
GEMM Specialization for
C = Add(GeMM(A, B) + bias, D0)),
where A[RowMajor][M, K], B[ColMajor][N, K], C[RowMajor][M, N]
bias[RowMajor][N], D0[RowMajor][M, N]
*/

namespace flashck {

std::map<std::string, std::shared_ptr<void>> GemmRCRBiasAddKernel::Init(const OperationKind&   op_kind,
                                                                        const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<GemmOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmRCRBiasAddKernel::GenKernelProfiler(const std::string&                               model_name,
                                        const std::unordered_map<std::string, std::any>& kernel_func_map,
                                        const std::string&                               folder_name)
{
    RCRLayout rcr_layout;

    return GenGemmCommonKernelProfiler(model_name, kernel_func_map, rcr_layout.GetGemmArgsParse(), "bias_add");
}

std::string GemmRCRBiasAddKernel::GenKernelFunction(const std::string&                               func_name,
                                                    const std::string&                               model_name,
                                                    const std::unordered_map<std::string, std::any>& kernel_func_map)
{

    return GenGemmCommonKernelFunction(func_name, kernel_func_map, "bias_add");
}

void GemmRCRBiasAddKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    auto gemm_args = std::get<GemmKernelArgs>(args);

    VLOG(1) << gemm_args.GetDimInfo();

    decltype(&GemmBiasElementwise) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    kernel_func(gemm_args.in_ptr_,
                gemm_args.weight_ptr_,
                gemm_args.out_ptr_,
                gemm_args.bias_ptr_,
                gemm_args.d0_ptr_,
                gemm_args.a_dim0_,
                gemm_args.a_dim1_,
                gemm_args.b_dim0_,
                gemm_args.b_dim1_,
                gemm_args.c_dim0_,
                gemm_args.c_dim1_,
                gemm_args.stream_);
}

}  // namespace flashck