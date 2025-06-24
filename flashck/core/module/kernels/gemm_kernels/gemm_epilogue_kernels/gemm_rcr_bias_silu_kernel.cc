#include "flashck/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_silu_kernel.h"

#include "flashck/core/module/kernels/gemm_kernels/layout.h"

/*
GEMM ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[m, n] = SILU(a[m, k] * b[n, k] + bias[n])
This is used for `torch.nn.functional.linear + silu`
*/

static const std::string g_extra_code_source = R"(
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {
struct AddSiLU
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;
    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = 1.0f / (1.0f + exp(-a)) * a;
    };
    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        const double a = x0 + x1;
        y              = 1.0 / (1.0 + exp(-a)) * a;
    };
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t a = x0 + x1;
        y              = type_convert<half_t>(1.0) / (type_convert<half_t>(1.0) + type_convert<half_t>(exp(ck::type_convert<float>(-a)))) * a;
    };
};
} // namespace element_wise
} // namespace tensor_operation
} // namespace ck

)";

namespace flashck {

std::map<std::string, std::shared_ptr<void>> GemmRCRBiasSiLUKernel::Init(const OperationKind&   op_kind,
                                                                         const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<GemmOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmRCRBiasSiLUKernel::GenKernelProfiler(const std::string&                               model_name,
                                         const std::unordered_map<std::string, std::any>& kernel_func_map,
                                         const std::string&                               folder_name)
{
    RCRLayout rcr_layout;

    jinja2::ValuesMap extra_code_value_map{{}};
    std::string       extra_code = TemplateLoadAndRender(g_extra_code_source, extra_code_value_map);

    return GenGemmCommonKernelProfiler(
        model_name, kernel_func_map, rcr_layout.GetGemmArgsParse(), "bias_silu", extra_code);
}

std::string GemmRCRBiasSiLUKernel::GenKernelFunction(const std::string&                               func_name,
                                                     const std::string&                               model_name,
                                                     const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    jinja2::ValuesMap extra_code_value_map{{}};
    std::string       extra_code = TemplateLoadAndRender(g_extra_code_source, extra_code_value_map);
    return GenGemmCommonKernelFunction(func_name, kernel_func_map, "bias_silu", extra_code);
}

void GemmRCRBiasSiLUKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    auto gemm_args = std::get<GemmKernelArgs>(args);

    decltype(&GemmBias) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    kernel_func(gemm_args.in_ptr_,
                gemm_args.weight_ptr_,
                gemm_args.out_ptr_,
                gemm_args.bias_ptr_,
                gemm_args.a_dim0_,
                gemm_args.a_dim1_,
                gemm_args.b_dim0_,
                gemm_args.b_dim1_,
                gemm_args.c_dim0_,
                gemm_args.c_dim1_,
                gemm_args.stream_);
}

}  // namespace flashck