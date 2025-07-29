#pragma once

#include <string>

/// @brief GEMM data type declaration template
static const std::string g_gemm_common_dtype_decl_tpl = R"(
using ADataType         = {{a_dtype}};
using BDataType         = {{b_dtype}};
using CDataType         = {{c_dtype}};
using AccDataType       = float;
using D0DataType = {{d0_dtype}};
using D1DataType = {{d1_dtype}};
)";

/// @brief GEMM layout declaration template
static const std::string g_gemm_common_layout_decl_tpl = R"(
using ALayout         = {{a_layout}};
using BLayout         = {{b_layout}};
using CLayout         = {{c_layout}};
using D0Layout = {{d0_layout}};
using D1Layout = {{d1_layout}};
)";

/// @brief Template for conditional code generation
static const std::string g_gemm_common_common_running_cond_tpl = R"(
    if ({{cond}}) {
        {{program}}
    }
)";

/// @brief Macro declarations for symbol visibility
static const std::string g_gemm_common_macro_decl = R"(
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

/// @brief Template for kernel instance declaration
static const std::string g_gemm_common_instance_tpl = R"(
{{instance_code}}
using {{instance_alias_name}} = {{instance_name}};
)";


/// @brief Template for kernel execution runtime code
static const std::string g_gemm_common_running_tpl = R"(
    {{make_args}}

    {{instance_decl}}

    float ave_time{0};

    const auto RunFunc = [&](){
        auto kargs   = Kernel::MakeKernelArgs(args);

{% if is_persistent %}
        const dim3 grids = Kernel::MaxOccupancyGridSize(s);
{% else %}
        const dim3 grids = {{instance_alias_name}}::GridSize(args);
{% endif %}
        constexpr dim3 blocks = {{instance_alias_name}}::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
            {
                throw ArgumentsNotSupportedException(
                    "Wrong! Arguments not supported! Skipping gemm!\n");
            }

{% if is_running %}
        auto s = ck_tile::stream_config{stream, {{is_profiling}}/*time_kernel*/};
{% else %}
        auto s = ck_tile::stream_config{stream, {{is_profiling}}/*time_kernel*/, 
            {{log_level}}/*log_level*/, {{cold_niters}}/*cold_niters*/, 
            {{nrepeat}}/*nrepeat*/, {{is_gpu_timer}}/*is_gpu_timer*/, 
            {{flush_cache}}/*flush_cache*/, {{rotating_count}}/*rotating_count*/};

{% if not is_running && flush_cache %}
        static constexpr ck_tile::index_t APackedSize =
                    std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
        static constexpr ck_tile::index_t BPackedSize =
            std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
        
        ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                    args.M, args.K, args.stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
            args.K, args.N, args.stride_B, is_row_major(BLayout{})));

        auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
        auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

        ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
            kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b_buffer);
        rotating_mem.Print();

        uto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
                };
        float ave_time = ck_tile::launch_kernel_preprocess(
            s,
            run_flush_cache,
            ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                {{instance_alias_name}}{}, grids, blocks, 0, kargs));

{% else %}
        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, {{block_per_cu}}>({{instance_alias_name}}{}, grids, blocks, 0, kargs));
{% endif %}
    }
  
    BaseGemmPipeline::TailHandler(RunFunc, has_hot_loop, tail_num);

    if(ave_time < 0) {
        std::cerr << {{instance_alias_name}}::GetName() << " not supported!\n" << std::flush;
    }

{% if not is_running %}
    std::size_t flop = std::size_t(2)  * m * n;
    num_byte += sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    std::cout << "KERNEL: " << "{{instance_name}}" << std::endl;
    std::cout << "LATENCY: " << ave_time << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << " Tflops" << std::endl;
    std::cout << "BANDWIDTH: " << gb_per_sec << " GB/s" << std::endl;

{% endif %}
)";

/// @brief Template for kernel function generation
static const std::string g_gemm_common_kernel_func_tpl = R"(
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include <string>

{{header}}

#include "ck_tile/host.hpp"
#include "norm_common.h"

{{macro_decl}}

{{dtype_decl}}

{% if is_running %} {{c_flag}} FC_EXPORT {% endif %} {{func_signature}} {
    {{execute_func}}
}
)";


/// @brief Template for profiling main function
static const std::string g_gemm_common_profiling_tpl = R"(
{{kernel_func}}

{{create_args}}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if (!result)
        return -1;

    {{arg_parser}}

    {{tensor_decl}}

    hipStream_t stream = nullptr;

    {{func_call}} 
}

)";

