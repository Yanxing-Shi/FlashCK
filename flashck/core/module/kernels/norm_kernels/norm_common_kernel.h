#pragma once

#include "flashck/core/module/kernels/kernel.h"
#include "flashck/core/module/kernels/kernel_registry.h"

static const std::string g_norm_exec_cond_tpl = R"(
    if ({{cond}}) {
        {{program}}
    }
)";

static const std::string g_norm_macro_decl = R"(
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

static const std::string g_norm_instance_tpl = R"(
{{instance_code}}
using {{instance_alias_name}} = {{config_name}};
)";

static const std::string g_norm_exec_tpl = R"(
    {{make_args}}

    const dim3                 grids       = {{instance_alias_name}}::GridSize(args);
    constexpr dim3             blocks      = {{instance_alias_name}}::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;
    auto                       s           = ck_tile::stream_config{stream, {{is_profiling}}, 0, 5/*warmup*/, 20/*repeat*/};
    auto kargs = {{instance_alias_name}}::MakeKargs(args);
    
    float ave_time =
        ck_tile::launch_kernel(s, ck_tile::make_kernel<blocks.x, kBlockPerCu>({{instance_alias_name}}{}, grids, blocks, 0, kargs));

    if(ave_time < 0)
    {
        std::cerr << {{instance_alias_name}}::GetName()<< " not supported !" << std::endl << std::flush;
    }

{% if not is_running %}
    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(GammaDataType) * n +
                           sizeof(BetaDataType) * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    // std::cout << "KERNEL:" << {{instance_alias_name}}::GetName() << ",";
    std::cout<< "KERNEL:" << "{{config_name}}" << ",";
    std::cout << "TIME:" << ave_time << "ms" << std::endl << std::flush;
    // std::cout << "GB_PER_SEC:" << gb_per_sec << "GB/s" << std::endl << std::flush;
{% endif %}
)";

static const std::string g_norm_kernel_func_tpl = R"(
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include <string>

#include "ck_tile/host.hpp"
#include <cstring>
#include <algorithm>

#include "norm_common.h"

{{macro_decl}}

{{dtype_decl}}

{{instances_decl}}

{% if is_running %} {{c_flag}} FC_EXPORT {% endif %} {{func_signature}} {
    {{execute_func}}
}
)";

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

namespace flashck {

class NormCommonKernel: public Kernel {
public:
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CommonCodeGenForTuning(const std::string&                                  model_name,
                           const Problem&                                      problem,
                           const std::map<std::string, std::unique_ptr<void>>& instance_map,
                           const TuningTpl&                                    tuning_tpl,
                           const std::string&                                  folder_name = "kernel_profile");

    // std::string CommonCodeGenForRunning(const std::string&                               func_name,
    //                                     const std::string&                               model_name,
    //                                     const std::unordered_map<std::string, std::any>& kernel_func_map,
    //                                     const RunningTpl&                                running_tpl);
};
}  // namespace flashck
