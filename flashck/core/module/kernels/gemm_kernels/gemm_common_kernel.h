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

#include "flashck/core/utils/jinjia2_utils.h"

#include "flashck/core/module/kernels/kernel.h"

#include "flashck/core/profiling/base.h"
#include "flashck/core/profiling/library.h"

static const std::string g_exec_cond_tpl = R"(
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}} } else {
{{indent}} std::cerr << "wrong! "<< "{{cond}}" << " does not support this Gemm instance." << std::endl;
{{indent}} return;
{{indent}} }
)";

static const std::string g_inline_utils_tpl = R"(
#include "host_tensor.cpp"
#include "device_memory.cpp"
)";

static const std::string g_extra_header_tpl = R"(
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

static const std::string g_macro_decl_tpl = R"(
// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with FC_EXPORT to make
// them visible.
#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define FC_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define FC_EXPORT __declspec(dllexport)
#else
#define FC_EXPORT
#endif
#endif
)";

static const std::string g_profiler_header_tpl = R"(
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "ck/library/utility/literals.hpp"
)";

static const std::string g_dtype_decl_tpl = R"(
using ADataType               = {{a_ck_dtype}};
using BDataType               = {{b_ck_dtype}};
using CDataType               = {{c_ck_dtype}};
{% if has_bias %}
using BiasDataType            = {{bias_ck_dtype}};
{% endif %}
{% if has_scale %}
using ScaleAElementType = {{scale_a_ck_dtype}};
using ScaleBElementType = {{scale_b_ck_dtype}};
{% endif %}
)";

static const std::string g_layout_decl_tpl = R"(
using ALayout  = {{a_layout}};
using BLayout  = {{b_layout}};
using CLayout  = {{c_layout}};
{% if has_bias %}
using BiasLayout = {{bias_layout}};
{% endif %}
)";

static const std::string g_instance_tpl = R"(
{{config}}
using {{name}} = {{config_name}};
)";

static const std::string g_exec_tpl = R"(
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

static const std::string g_problem_args_tpl = R"(
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
{% if is_running %}
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

static const std::string g_extra_shape_tpl = R"(
{{indent}}ck::index_t stride_a = a_dim1;
{{indent}}ck::index_t stride_b = b_dim1;
{{indent}}ck::index_t stride_c = c_dim1;
)";

static const std::string g_src_tpl = R"(
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

{% if is_running %} {{c_flag}} FC_EXPORT {% endif %} void {{function_name}}(
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
{% if "split_k" in gemm_flag and is_running == False %}
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

static const std::string g_structs_tpl = R"(
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

const static std::string g_func_call_tpl = R"(
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
{% if "split_k" in gemm_flag and is_running == False %}
{{indent}}    split_k,
{% endif %}
{{indent}}    stream
{{indent}});
)";

static const std::string g_tensor_decl_tpl = R"(
{% if is_batched %}
    using strides_t = std::array<int32_t, 3>;
    auto get_strides = [](int32_t batch_stride, int32_t leading_dimension, auto layout) constexpr -> strides_t {
        if constexpr (std::is_same_v<decltype(layout), Row>) {
            return {batch_stride, leading_dimension, 1};
        }
        return {batch_stride, 1, leading_dimension};
    };
    auto a_size = strides_t{B, M, K};
    auto a_stride = get_strides(M * K, LDA, ALayout{});
    auto b_size = strides_t{B, N, K};
    auto b_stride = get_strides(N * K, LDB, BLayout{});
    auto c_size = strides_t{B, M, N};
    auto c_stride = get_strides(M * N, LDC, CLayout{});
    {% else %}
    using strides_t = std::array<int32_t, 2>;
    auto get_strides = [](int32_t leading_dimension, auto layout) constexpr -> strides_t {
        if constexpr (std::is_same_v<decltype(layout), Row>) {
            return {leading_dimension, 1};
        }
        return {1, leading_dimension};
    };
    auto a_size = strides_t{M, K};
    auto a_stride = get_strides(LDA, ALayout{});
    auto b_size = strides_t{N, K};
    auto b_stride = get_strides(LDB, BLayout{});
    auto c_size = strides_t{M, N};
    auto c_stride = get_strides(LDC, CLayout{});
{% endif %}

    Tensor<AElementType> a_m_k ( HostTensorDescriptor ( a_size, a_stride ) );
    Tensor<BElementType> b_k_n ( HostTensorDescriptor ( b_size, b_stride ) );
{% if has_bias %}
    Tensor<BiasElementType> d_m_n ( HostTensorDescriptor ( c_size, get_strides(LDD, BiasLayout{}) ) );
{% endif %}
{% if has_scale %}
    // NB: these are hardcoded
    Tensor<ScaleAElementType> s_a_m_n ( HostTensorDescriptor ( strides_t{M, N}, get_strides(0, Row{}) ));
    Tensor<ScaleAElementType> s_b_m_n ( HostTensorDescriptor ( strides_t{M, N}, get_strides(0, Col{}) ));
{% endif %}

    Tensor<CElementType> c_m_n_host ( HostTensorDescriptor ( c_size, c_stride ) );
    Tensor<CElementType> c_m_n_device ( HostTensorDescriptor ( c_size, c_stride ) );

    a_m_k.GenerateTensorValue(GeneratorTensor_2<AElementType>());
    b_k_n.GenerateTensorValue(GeneratorTensor_2<BElementType>());
    {% if has_bias %}
    d_m_n.GenerateTensorValue(GeneratorTensor_2<BiasElementType>());
    {% endif %}
    {% if has_scale %}
    s_a_m_n.GenerateTensorValue(GeneratorTensor_2<ScaleAElementType>());
    s_b_m_n.GenerateTensorValue(GeneratorTensor_2<ScaleBElementType>());
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

    DeviceMem a_m_k_device_buf(sizeof(AElementType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BElementType) * b_k_n.mDesc.GetElementSpaceSize());
    {% if has_bias %}
    DeviceMem d_m_n_device_buf(sizeof(BiasElementType) * d_m_n.mDesc.GetElementSpaceSize());
    {% endif %}
    {% if has_scale %}
    DeviceMem s_a_m_n_device_buf(sizeof(ScaleAElementType) * s_a_m_n.mDesc.GetElementSpaceSize());
    DeviceMem s_b_m_n_device_buf(sizeof(ScaleBElementType) * s_b_m_n.mDesc.GetElementSpaceSize());
    {% endif %}
    DeviceMem c_m_n_device_buf(sizeof(CElementType) * c_m_n_device.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    {% if has_bias %}
    d_m_n_device_buf.ToDevice(d_m_n.mData.data());
    {% endif %}
    {% if has_scale %}
    s_a_m_n_device_buf.ToDevice(s_a_m_n.mData.data());
    s_b_m_n_device_buf.ToDevice(s_b_m_n.mData.data());
    {% endif %}
)";

static const std::string g_profiler_tpl = R"(
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

const static std::string g_shape_eval_tpl = R"(
{{indent}}{{dtype}} {{name}} = {{dim_calculator}};
)";

namespace flashck {

class GemmCommonKernel: public Kernel {
public:
    GemmCommonKernel()          = default;
    virtual ~GemmCommonKernel() = default;

    std::map<std::string, std::shared_ptr<void>> ExtractConfig(const GemmOperationKind& op_kind,
                                                               const TensorOperation&   extra_kind);

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
                                const std::string& extra_shape_template                     = g_extra_shape_tpl,
                                const std::string& problem_args_template                    = g_problem_args_tpl,
                                const std::string& extra_header_template                    = g_extra_header_tpl,
                                const std::string& tensor_decl_template                     = g_tensor_decl_tpl,
                                const std::string& inverse_shape                            = "",
                                const std::string& input_addr_calculator                    = "",
                                const std::string& output_addr_calculator                   = "",
                                const std::string& folder_name                              = "kernel_profile");

    std::string GenGemmCommonKernelFunction(const std::string&                               func_name,
                                            const std::unordered_map<std::string, std::any>& kernel_func_map,
                                            const std::string&                               gemm_flag  = "",
                                            const std::string&                               extra_code = "",
                                            const int                                        ndims      = 2,
                                            const std::string& extra_shape_template  = g_extra_shape_tpl,
                                            const std::string& problem_args_template = g_problem_args_tpl,
                                            const std::string& extra_header_template = g_extra_header_tpl,
                                            const std::string& inverse_shape         = "");

    void FilterInstance() {}
};
}  // namespace flashck