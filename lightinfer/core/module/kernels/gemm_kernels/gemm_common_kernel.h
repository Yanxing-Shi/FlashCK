#pragma once

#include <any>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "lightinfer/core/utils/jinjia2_utils.h"

#include "lightinfer/core/module/kernels/kernel.h"

#include "lightinfer/core/profiler/base.h"
#include "lightinfer/core/profiler/library.h"

static const std::string g_exec_cond_source = R"(
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}} } else {
{{indent}} std::cerr << "wrong! "<< "{{cond}}" << " does not support this Gemm instance." << std::endl;
{{indent}} return;
{{indent}} }
)";

static const std::string g_extra_header_source = R"(
{% if gemm_flag == "" %}
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
{% elif gemm_flag == "permute_m2n3" %}
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
{% elif "bias" in gemm_flag or has_d0 %}
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
    {% if gemm_flag in ["bias_permute_m2n3", "bias_permute_m3n2"]  %}
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
    {% endif %}
{% elif "split_k" in gemm_flag %}
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"
{% endif %}
)";

static const std::string g_macro_decl_source = R"(
// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with ATER_EXPORT to make
// them visible.
#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define ATER_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define ATER_EXPORT __declspec(dllexport)
#else
#define ATER_EXPORT
#endif
#endif
)";

static const std::string g_profiler_header_source = R"(
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "ck/library/utility/literals.hpp"
)";

static const std::string g_dtype_decl_source = R"(
using ADataType               = {{a_dtype}};
using BDataType               = {{b_dtype}};
using CDataType               = {{c_dtype}};
)";

static const std::string g_layout_decl_source = R"(
using ALayout  = {{a_layout}};
using BLayout  = {{b_layout}};
using CLayout  = {{c_layout}};
)";

static const std::string g_instance_source = R"(
{{config}}
using {{name}} = {{config_name}};
)";

static const std::string g_exec_source = R"(
{{indent}}auto device_instance =  {{instance}}{};
{{indent}}auto argument = device_instance.MakeArgument(
{{problem_args}}
{{indent}});
{{indent}}if(!device_instance.IsSupportedArgument(argument)) {
{{indent}}  std::cerr << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Gemm problem.";
{{indent}} return;}
{{indent}}auto invoker  = device_instance.MakeInvoker();
{{indent}}invoker.Run(argument, StreamConfig{stream, false});
{{indent}}
)";

static const std::string g_problem_args_source = R"(
{{indent}}                                static_cast<ADataType*>(in_ptr),
{{indent}}                                static_cast<BDataType*>(weight_ptr),
{% if gemm_flag == "bias_permute" %}
{{indent}}                                static_cast<CDataType*>(bias_ptr),
{% elif gemm_flag == "permute" %}
{{indent}}                                nullptr,
{% elif gemm_flag == "bias_permute_m2n3" %}
{{indent}}                                std::array<const void*, 1>{static_cast<CDataType*>(bias_ptr)},
{% elif gemm_flag == "permute_m2n3" %}
{{indent}}                                {},
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<const void*, 1>{static_cast<CDataType*>(bias_ptr)},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<const void*, 2>{static_cast<CDataType*>(bias_ptr),
                                                                    static_cast<CDataType*>(d0_ptr)},
{% elif has_d1 %}
{{indent}}                                std::array<const void*, 3>{static_cast<CDataType*>(bias_ptr),
                                                                    static_cast<CDataType*>(d0_ptr),
                                                                    static_cast<CDataType*>(d1_ptr)},
{% endif %}
{% endif %}
{{indent}}                                static_cast<CDataType*>(out_ptr),
{% if gemm_flag in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"] %}
{% else %}
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{% if "split_k" in gemm_flag %}
{% if is_execute %}
{{indent}}                                {{split_k}},
{% else %}
{{indent}}                                split_k,
{% endif %}
{% endif %}
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% endif %}
{% if gemm_flag == "bias_permute" %}
{{indent}}                                {M0, M1, M2, N0, N1, stride_D_M0, stride_D_M1, stride_D_M2, stride_D_N0, stride_D_N1},
{{indent}}                                {M0, M1, M2, N0, N1, stride_E_M0, stride_E_M1, stride_E_M2, stride_E_N0, stride_E_N1},
{% elif gemm_flag == "permute" %}
{{indent}}                                {},
{{indent}}                                {M0, M1, M2, N0, N1, stride_E_M0, stride_E_M1, stride_E_M2, stride_E_N0, stride_E_N1},
{% elif gemm_flag in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"]  %}
{{indent}}                                a_ms_ks_lengths,
{{indent}}                                a_ms_ks_strides,
{{indent}}                                b_ns_ks_lengths,
{{indent}}                                b_ns_ks_strides,
    {% if gemm_flag == "permute_m2n3"  %}
{{indent}}                                {},
{{indent}}                                {},
    {% else %}
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
    {% endif %}
{{indent}}                                e_ms_ns_lengths,
{{indent}}                                e_ms_ns_strides,
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<ck::index_t, 2>{0, static_cast<int>(stride_c)},
{% elif has_d1 %}
{{indent}}                                std::array<ck::index_t, 3>{0, static_cast<int>(stride_c), static_cast<int>(stride_c)},
{% endif %}
{{indent}}                                stride_c,
{% endif %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" or gemm_flag == "split_k" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag in ["permute", "permute_m2n3"] %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "bias" or "bias_permute" in gemm_flag %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif gemm_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif gemm_flag == "bias_gelu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddFastGelu{}
{% elif gemm_flag == "bias_swish" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSwish{}
{% elif gemm_flag == "bias_hardswish" %}
{{indent}}                                ck::tensor_operation::element_wise::AddHardswish{}
{% elif gemm_flag == "bias_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddTanh{}
{% elif gemm_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% elif gemm_flag == "bias_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAdd{}
{% elif gemm_flag == "bias_mul" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMultiply{}
{% elif gemm_flag == "bias_silu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSiLU{}
{% endif %}
)";

static const std::string g_extra_shape_source = R"(
{{indent}}ck::index_t stride_a = a_dim1;
{{indent}}ck::index_t stride_b = b_dim1;
{{indent}}ck::index_t stride_c = c_dim1;
)";

static const std::string g_src_source = R"(
#include "ck/ck.hpp"

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

{{macro_decl}}

{{extra_header}}

{{profiler_header}}

{{dtype_decl}}

{{layout_decl}}

{{extra_code}}

{{instances}}

{{structs_def}}

{% if is_execute %} {{c_flag}} ATER_EXPORT {% endif %} void {{function_name}}(
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,
{% if "bias" in gemm_flag or gemm_flag == "add" %}
    void* bias_ptr,
{% endif %}
{% if has_d0 %}
    void* d0_ptr,
{% endif %}
{% if has_d1 %}
    void* d1_ptr,
{% endif %}
{% for idx in range(ndims) %}
    int64_t a_dim{{idx}},
{% endfor %}
{% for idx in range(ndims) %}
    int64_t b_dim{{idx}},
{% endfor %}
{% if gemm_flag != "bias_b1"%}
{% for idx in range(ndims) %}
    int64_t c_dim{{idx}},
{% endfor %}
{%else%}
{% for idx in range(ndims) %}
    int64_t b1_dim{{idx}},
{% endfor %}
{%endif%}
{% for idx in range(p_dims) %}
    int64_t p_dim{{idx}},
{% endfor %}
{% if "split_k" in gemm_flag and is_execute == False %}
    int64_t split_k,
{% endif %}
    hipStream_t stream
    ) {

    {{shape_func}}

    {{inverse_shape}}

    {{extra_shape}}

    {{input_addr_calculator}}

    {{output_addr_calculator}}

    {{exec_paths}}
}

)";

static const std::string g_structs_source = R"(
constexpr int error_exit_code = -1;

#define HIP_CHECK(condition)                                                                                           \
    {                                                                                                                  \
        const hipError_t error = condition;                                                                            \
        if (error != hipSuccess) {                                                                                     \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " << __FILE__ << ':'         \
                      << __LINE__ << std::endl;                                                                        \
            std::exit(error_exit_code);                                                                                \
        }                                                                                                              \
    }

class HipTimer {
private:
    hipEvent_t  event_start_;
    hipEvent_t  event_stop_;
    hipStream_t stream_;

public:
    explicit HipTimer(hipStream_t stream = 0)
    {
        stream_ = stream;
    }
    void Start()
    {
        HIP_CHECK(hipEventCreate(&event_start_));
        HIP_CHECK(hipEventCreate(&event_stop_));
        HIP_CHECK(hipEventRecord(event_start_, stream_));
    }
    float Stop()
    {
        float time;
        HIP_CHECK(hipEventRecord(event_stop_, stream_));
        HIP_CHECK(hipEventSynchronize(event_stop_));
        HIP_CHECK(hipEventElapsedTime(&time, event_start_, event_stop_));
        HIP_CHECK(hipEventDestroy(event_start_));
        HIP_CHECK(hipEventDestroy(event_stop_));
        return time;
    }
    ~HipTimer() {}
};
)";

const static std::string g_func_call_source = R"(
{{indent}}{{func_name}}(
{{indent}}    in_dev_buff_ptr,
{{indent}}    weight_dev_buff_ptr,
{{indent}}    out_dev_buff_ptr,
{% if "bias" in gemm_flag or gemm_flag == "add" %}
{{indent}}    bias_dev_buff_ptr,
{% endif %}
{% if has_d0 %}
{{indent}}    d0_dev_buff_ptr,
{% endif %}
{% if has_d1 %}
{{indent}}    d1_dev_buff_ptr,
{% endif %}
{% for dim in a_dims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in b_dims %}
{{indent}}    {{dim}},
{% endfor %}
{% if gemm_flag != "bias_b1" %} 
{% for dim in c_dims %}
{{indent}}    {{dim}},
{% endfor %}
{% else %}
{% for dim in b1_dims %}
{{indent}}    {{dim}},
{% endfor %}
{% endif %}
{% for dim in p_dims %}
{{indent}}    {{dim}},
{% endfor %}
{% if "split_k" in gemm_flag and is_execute == False %}
{{indent}}    split_k,
{% endif %}
{{indent}}    stream
{{indent}});
)";

static const std::string g_tensor_decl_source = R"(
    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a(f_host_tensor_descriptor(M, K, stride_a, ALayout{}));
    Tensor<BDataType> b(f_host_tensor_descriptor(K, N, stride_b, BLayout{}));
    Tensor<CDataType> c(f_host_tensor_descriptor(M, N, stride_c, CLayout{}));

{% if "bias" in gemm_flag %}
   Tensor<CDataType> d(f_host_tensor_descriptor(M, N, 0, CLayout{}));
{% endif %}
{% if has_d0 %}
    Tensor<CDataType> d0(f_host_tensor_descriptor(M, N, stride_c, CLayout{}));
{% endif %}
{% if has_d1 %}
    Tensor<CDataType> d1(f_host_tensor_descriptor(M, N, stride_c, CLayout{}));
{% endif %}

    a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    b.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    c.GenerateTensorValue(GeneratorTensor_3<CDataType>{-0.5, 0.5});
{% if "bias" in gemm_flag %}
    d.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
{% endif %}
{% if has_d0 %}
    d0.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
{% endif %}
{% if has_d1 %}
    d1.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
{% endif %}

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c.mDesc.GetElementSpaceSize());
{% if "bias" in gemm_flag %}
    DeviceMem d_device_buf(sizeof(CDataType) * d.mDesc.GetElementSpaceSize());
{% endif %}
{% if has_d0 %}
    DeviceMem d0_device_buf(sizeof(CDataType) * d0.mDesc.GetElementSpaceSize());
{% endif %}
{% if has_d1 %}
    DeviceMem d1_device_buf(sizeof(CDataType) * d1.mDesc.GetElementSpaceSize());
{% endif %}

    a_device_buf.ToDevice(a.mData.data());
    b_device_buf.ToDevice(b.mData.data());
    c_device_buf.ToDevice(c.mData.data());
{% if "bias" in gemm_flag %}
    d_device_buf.ToDevice(d.mData.data());
{% endif %}
{% if has_d0 %}
    d0_device_buf.ToDevice(d0.mData.data());
{% endif %}
{% if has_d1 %}
    d1_device_buf.ToDevice(d1.mData.data());
{% endif %}

    auto in_dev_buff_ptr = a_device_buf.GetDeviceBuffer();
    auto weight_dev_buff_ptr = b_device_buf.GetDeviceBuffer();
    auto out_dev_buff_ptr = c_device_buf.GetDeviceBuffer();
{% if "bias" in gemm_flag %}
    auto bias_dev_buff_ptr = d_device_buf.GetDeviceBuffer();
{% endif %}
{% if has_d0 %}
    auto d0_dev_buff_ptr = d0_device_buf.GetDeviceBuffer();
{% endif %}
{% if has_d1 %}
    auto d1_dev_buff_ptr = d1_device_buf.GetDeviceBuffer();
{% endif %}
)";

static const std::string g_profiler_source = R"(
{{op_func}}

{{structs_def}}

int main(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error("wrong params");
    }
    {{args_parse}}

    {{extra_shape}}

    hipStream_t stream = nullptr;
    constexpr int warm_iter    = 3;
    constexpr int profile_iter = 5;
    
    {{tensor_decl}}

    // warm up
    for(int i = 0; i < warm_iter; ++i) {
        {{func_call}}
    }

    // run
    HipTimer hip_timer;
    hip_timer.Start();
    for(int i = 0; i < profile_iter; ++i) {
        {{func_call}}
    }

    float  ave_time  = hip_timer.Stop() / profile_iter;

    std::cout << "KERNEL:" << "{{kernel_config_name}}" << ",";
    std::cout << "TIME:" << ave_time << "ms";
 
    return 0;
}
)";

const static std::string g_shape_eval_source = R"(
{{indent}}{{dtype}} {{name}} = {{dim_calculator}};
)";

namespace lightinfer {

class GemmCommonKernel: public Kernel {
public:
    GemmCommonKernel()          = default;
    virtual ~GemmCommonKernel() = default;

    /*
    Extract (operation name, operation instance) pair
     from all operation candidates.

     Parameters
     ----------
     op_kind : ck_lib.library.GemmOperationKind
         Operation kind.
     extra_kind : ck_lib.library.[AnyKind]
         Used to as extra flag to distinguish kernels.
         E.g. bias_add_relu vs. add_relu_bias
     f_prop_op: function
         Used to filter operation.

     Returns
     -------
     Dict
         Extracted (operation name, operation instance) pair.
    */
    std::map<std::string, std::shared_ptr<void>> ExtractConfig(const GemmOperationKind& op_kind,
                                                               const TensorOperation&   extra_kind);

    // Exract name from the statement, .g. 'model' for 'using model = xxx'.
    // std::string ExetractConfigName(const std::string& config);

    std::string GenDimCalculator(const std::shared_ptr<DimInfo>& dim_info, bool is_ptr);

    std::string GenShapeEvalCode(const std::string&                                                  dtype,
                                 const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
                                 bool                                                                is_ptr);

    /*
    Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    args_parse: str
        Profiler input argument parser.
    gemm_flag : str
        Flag telling which backend should be generated. options are
    '','bias','bias_relu','bias_sigmoid','bias_add_relu'. extra_code : str Extra code for self-defined operators.
    ndims : int Number of dims for each parameter, 2 for gemm, 3 for bmm extra_shape_template: jinja2.Template Shape
    evaluation template. problem_args_template: jinja2.Template Problem args template for profiler.
    extra_header_template: jinja2.Template
        Extra header template as we have different headers for gemm and bmm.
    tensor_decl_template: jinja2.Template
        Tensor declaration template.
    */
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

    // void GemmCommonKernelLauncher(const std::string&        kernel_func_name,
    //                               const GemmKernelArgs&     args,
    //                               const GemmKernelCallType& kernel_call_type);
};
}  // namespace lightinfer