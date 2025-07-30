#pragma once

#include <string>


/// @brief Template for conditional code generation
static const std::string g_norm_running_cond_tpl = R"(
    if ({{cond}}) {
        {{program}}
    } else {
        std::cerr << "wrong! "<< "{{cond}}" << " does not support this Gemm instance." << std::endl;
        return;
    }
)";

/// @brief Macro declarations for symbol visibility
static const std::string g_norm_macro_decl = R"(
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

/// @brief Template for creating argument parser
static const std::string g_norm_create_args_tpl = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("e", "1e-5", "epsilon")
        .insert("x_stride", "-1", "x row_stride, if -1 then equal to n")
        .insert("xr_stride", "-1", "x residule row_stride, if -1 then equal to n")
        .insert("y_stride", "-1", "y row_stride, if -1 then equal to n")
        .insert("yr_stride", "-1", "y residule row_stride, if -1 then equal to n");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

/// @brief Template for kernel instance declaration
static const std::string g_norm_instance_tpl = R"(
{{instance_code}}
using {{instance_alias_name}} = {{instance_name}};
)";

/// @brief Template for kernel execution runtime code
static const std::string g_norm_running_tpl = R"(
    {{make_args}}

    const dim3 grids = {{instance_alias_name}}::GridSize(args);
    constexpr dim3 blocks = {{instance_alias_name}}::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;
{% if is_running %}
    auto s = ck_tile::stream_config{stream, {{is_profiling}}/*time_kernel*/};
{% else %}
    auto s = ck_tile::stream_config{stream, {{is_profiling}}/*time_kernel*/, 
            {{log_level}}/*log_level*/, {{cold_niters}}/*cold_niters*/, 
            {{nrepeat}}/*nrepeat*/, {{is_gpu_timer}}/*is_gpu_timer*/, 
            {{flush_cache}}/*flush_cache*/, {{rotating_count}}/*rotating_count*/};
{% endif %}
    auto kargs = {{instance_alias_name}}::MakeKargs(args);
    
    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>({{instance_alias_name}}{}, grids, blocks, 0, kargs));

    if(ave_time < 0) {
        std::cerr << {{instance_alias_name}}::GetName() << " not supported!\n" << std::flush;
    }

{% if not is_running %}
    std::size_t flop = std::size_t(2)  * m * n;
    {% if kind == "layer_norm" %}
    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(GammaDataType) * n +
                           sizeof(BetaDataType) * n + sizeof(YDataType) * m * n;
    {% else %}
    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(GammaDataType) * n +
                           sizeof(YDataType) * m * n;
    {% endif %}
    float gb_per_sec = num_byte / 1.E6 / ave_time;
    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;
    
    std::cout << "KERNEL: " << "{{instance_name}}" << std::endl;
    std::cout << "LATENCY: " << ave_time << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << " Tflops" << std::endl;
    std::cout << "BANDWIDTH: " << gb_per_sec << " GB/s" << std::endl;

{% endif %}
)";

/// @brief Template for kernel function generation
static const std::string g_norm_kernel_func_tpl = R"(
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
{% if kind == "layer_norm" %}
#include "ck_tile/ops/layernorm2d.hpp"
{% else %}
#include "ck_tile/ops/rmsnorm2d.hpp"
{% endif %}
#include <ck_tile/ops/epilogue.hpp>
#include <string>

#include "ck_tile/host.hpp"
#include <cstring>
#include <algorithm>

#include "norm_common.h"

{{macro_decl}}

{{dtype_decl}}

{{instance_decl}}

{% if is_running %} {{c_flag}} FC_EXPORT {% endif %} {{func_signature}} {
    {{execute_func}}
}
)";

/// @brief Template for profiling main function
static const std::string g_norm_profiling_tpl = R"(
{{kernel_func}}

{{create_args}}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if (!result)
        return -1;

    ck_tile::index_t m        = arg_parser.get_int("m");
    ck_tile::index_t n        = arg_parser.get_int("n");
    float epsilon       = arg_parser.get_float("e");
    ck_tile::index_t x_stride = arg_parser.get_int("x_stride");
    if (x_stride < 0)
        x_stride = n;
    ck_tile::index_t xr_stride = arg_parser.get_int("xr_stride");
    if (xr_stride < 0)
        xr_stride = n;
    ck_tile::index_t y_stride = arg_parser.get_int("y_stride");
    if (y_stride < 0)
        y_stride = n;
    ck_tile::index_t yr_stride = arg_parser.get_int("yr_stride");
    if (yr_stride < 0)
        yr_stride = n;

    assert(x_stride >= n);

    {{tensor_decl}}

    hipStream_t stream = nullptr;

    {{func_call}}  
}

)";
