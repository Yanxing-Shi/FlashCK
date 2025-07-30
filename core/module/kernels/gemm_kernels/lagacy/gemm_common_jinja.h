#pragma once

#include <string>

static const std::string g_legacy_gemm_exec_cond_tpl = R"(
    if ({{cond}}) {
        {{program}}
    } else {
        std::cerr << "wrong! "<< "{{cond}}" << " does not support this Gemm instance." << std::endl;
        return;
    }
)";

static const std::string g_legacy_gemm_macro_decl_tpl = R"(
// Symbol visibility macros
#ifdef __GNUC__
#define FC_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define FC_EXPORT __declspec(dllexport)
#else
#define FC_EXPORT
#endif
#endif
)";

static const std::string g_legacy_gemm_profiler_header_tpl = R"(
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "ck/library/utility/literals.hpp"
)";

static const std::string g_legacy_gemm_dtype_decl_tpl = R"(
using ADataType               = {{a_dtype}};
using BDataType               = {{b_dtype}};
using CDataType               = {{c_dtype}};
)";

static const std::string g_legacy_gemm_layout_decl_tpl = R"(
using ALayout  = {{a_layout}};
using BLayout  = {{b_layout}};
using CLayout  = {{c_layout}};
)";

static const std::string g_legacy_gemm_instance_tpl = R"(
{{instance_code}}
using {{instance_alias_name}} = {{instance_name}};
)";

static const std::string g_legacy_gemm_running_tpl = R"(
    auto device_instance =  {{instance_alias_name}}{};
    auto argument = device_instance.MakeArgument(
        {{problem_args}}
    );
    if(!device_instance.IsSupportedArgument(argument)) {
        std::cerr << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Gemm problem.";
        return;
    }
    auto invoker  = device_instance.MakeInvoker();
{% if is_running %}
    auto s = ck::stream_config{stream, {{is_profiling}}/*time_kernel*/};
{% else %}
    auto s = ck::stream_config{stream, {{is_profiling}}/*time_kernel*/, 
            {{log_level}}/*log_level*/, {{cold_niters}}/*cold_niters*/, 
            {{nrepeat}}/*nrepeat*/, {{is_gpu_timer}}/*is_gpu_timer*/, 
            {{flush_cache}}/*flush_cache*/, {{rotating_count}}/*rotating_count*/};
{% endif %}

    float ave_time = invoker.Run(argument, s);

    if(ave_time < 0) {
        std::cerr << {{instance_alias_name}}::GetName() << " not supported!\n" << std::flush;
    }

{% if not is_running %}
    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;
    float gb_per_sec = num_byte / 1.E6 / ave_time;
    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    std::cout << "KERNEL: " << "{{instance_name}}" << std::endl;
    std::cout << "LATENCY: " << ave_time << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << " Tflops" << std::endl;
    std::cout << "BANDWIDTH: " << gb_per_sec << " GB/s" << std::endl;
{% endif %}

)";

static const std::string g_legacy_gemm_extra_shape_tpl = R"(
    ck::index_t stride_a = a_dim1;
    ck::index_t stride_b = b_dim1;
    ck::index_t stride_c = c_dim1;
)";

static const std::string g_legacy_gemm_kernel_func_tpl = R"(
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

{% if is_execute %} {{c_flag}} ATER_EXPORT {% endif %} void {{func_signature}}
{
    {{shape_func}}

    {{inverse_shape}}

    {{extra_shape}}

    {{input_addr_calculator}}

    {{output_addr_calculator}}

    {{running_func}}
}

)";

static const std::string g_legacy_gemm_profiling_tpl = R"(
{{kernel_func}}

int main(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error("wrong params");
    }
    {{args_parse}}

    {{extra_shape}}

    hipStream_t stream = nullptr;

    {{tensor_decl}};
 
    return 0;
}
)";

const static std::string g_legacy_gemm_shape_eval_tpl = R"(
{{indent}}{{dtype}} {{name}} = {{dim_calculator}};
)";