#pragma once

#include "flashck/core/profiling/base.h"

#include "flashck/core/module/kernels/fmha_kernels/fmha_kernel_static.h"
#include "flashck/core/module/kernels/kernel.h"
#include "flashck/core/module/kernels/kernel_registry.h"

static const std::string g_fmha_exec_cond_source = R"(
    if ({{cond}}) {
        {{program}}
    }
)";

static const std::string g_fmha_macro_decl = R"(
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

static const std::string g_fmha_dtype_decl_source = R"(
using TypeConfig = FmhaFwdTypeConfig<{{DataType}}>;

using QDataType             = typename TypeConfig::QDataType;
using KDataType             = typename TypeConfig::KDataType;
using VDataType             = typename TypeConfig::VDataType;
using BiasDataType          = typename TypeConfig::BiasDataType;
using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
using LSEDataType           = typename TypeConfig::LSEDataType;
using SaccDataType          = typename TypeConfig::SaccDataType;
using SMPLComputeDataType   = typename TypeConfig::SMPLComputeDataType;
using PDataType             = typename TypeConfig::PDataType;
using OaccDataType          = typename TypeConfig::OaccDataType;
using ODataType             = typename TypeConfig::ODataType;
)";

static const std::string g_fmha_instance_source = R"(
{{config}}
using {{kernel_name}} = {{config_name}};
)";

static const std::string g_fmha_execute_source = R"(
    {{prepare_args}}

    constexpr dim3 blocks             = {{kernel_name}}::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = {{kernel_name}}::kBlockPerCu;
    auto                       s           = ck_tile::stream_config{stream, {{is_profile_kernel}}, 0, 5/*warmup*/, 20/*repeat*/};

    {{make_args}}

    float ave_time = ck_tile::launch_kernel(s, ck_tile::make_kernel<blocks.x, kBlockPerCu>({{kernel_name}}{}, grids, blocks, 0, kargs));
    if(ave_time < 0)
    {
        std::cerr << {{kernel_name}}::GetName()<< " not supported !" << std::endl << std::flush;
    }

{% if not is_execute %}
    // float tflops = static_cast<float>(flop) / 1.E9 / ave_time;
    // float gb_per_sec = num_byte / 1.E6 / ave_time;
    // std::cout << "KERNEL:" << {{kernel_name}}::GetName() << ",";
    std::cout<< "KERNEL:" << "{{config_name}}" << ",";
    std::cout << "TIME:" << ave_time << "ms" << std::endl << std::flush;
    // std::cout << "GB_PER_SEC:" << gb_per_sec << "GB/s" << std::endl << std::flush;
{% endif %}

)";

static const std::string g_fmha_kernel_func = R"(
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"

#include "ck_tile/host.hpp"
#include <cstring>
#include <algorithm>

#include "fmha_{{fmha_flag}}_common.h"

{{macro_decl}}

{{dtype_decl}}

{{instances_decl}}

{{func_signature}}
{
    {{execute_func}}
}
)";

const static std::string g_fmha_tenosr_decl_source = R"(
   // host memory for storing all the tensor elements
    auto [seqlen_qs, seqlen_ks, seqlen_kpads] =
        decode_seqlen({% if mode_str == "batch" %} mode_enum::batch,
                      {% else %} mode_enum::group,
                      {% endif %}
                      batch,
                      std::to_string(seqlen_q), // seqlen_q. if group-mode, means the average value of seqlen_q 
                      std::to_string({% if fmha_flag == "fwd_appendkv" %} seqlen_k + seqlen_knew {% elif fmha_flag == "fwd" or fmha_flag == "fwd_splitkv" %} seqlen_k {% else %} 1 {% endif %}), // seqlen_k (including new key/value), -1 means equal to s
                      std::to_string(-1), // seqlen_k stride between 2 tokens, currently used in group-mode only
                      {% if fmha_flag == "fwd_appendkv" %} seqlen_knew, {% else %} 0, {% endif %}
                      {% if fmha_flag == "fwd_appendkv" %} true {% else %} false {% endif %}
                      );

    const auto seqstart_q_host              = to_seqstarts(seqlen_qs);
    const auto seqstart_k_host              = to_seqstarts(seqlen_ks);
    const auto seqstart_k_with_padding_host = to_seqstarts(seqlen_kpads);

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<ck_tile::index_t>::min(); // we will use max seqlen to decide grid size
    auto max_seqlen_k = std::numeric_limits<ck_tile::index_t>::min();
    
    for(ck_tile::index_t wb = 0; wb < batch; ++wb)
    {
        const ck_tile::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck_tile::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        if(max_seqlen_q < real_seqlen_q)
        {
            max_seqlen_q = real_seqlen_q;
        }

        if(max_seqlen_k < real_seqlen_k)
        {
            max_seqlen_k = real_seqlen_k;
        }
        
    }

    const ck_tile::index_t shape_batch = {% if mode_str == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t shape_seqlen_q = {% if mode_str == "batch" %} seqlen_qs[0]; {% else %} seqstart_q_host.back(); {% endif %}
    const ck_tile::index_t shape_seqlen_k = {% if mode_str == "batch" %} seqlen_ks[0]; {% else %} seqlen_kpads[0] < 0 ?  seqstart_k_host.back(): seqstart_k_with_padding_host.back(); {% endif %}

    {{decl_source}}
)";

static const std::string g_fmha_profiler_source = R"(
{{kernel_func}}

{{create_args}}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if (!result)
        return -1;

    {{args_parser}}

    {{tensor_decl}}

    {{tensor_generate}}

    hipStream_t stream = nullptr;

    {{func_call}}

}

)";

namespace flashck {

class FmhaCommonKernel: public Kernel {
public:
    std::map<std::string, std::shared_ptr<void>> ExtractConfig(const FmhaOperationKind& op_kind,
                                                               const TensorOperation&   extra_kind);

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenFmhaCommonKernelProfiler(const std::string&                               model_name,
                                const std::unordered_map<std::string, std::any>& kernel_func_map,
                                const std::string&                               create_args_source,
                                const std::string&                               args_parser_source,
                                const std::string&                               args_decl_source,
                                const std::string&                               func_signature_source,
                                const std::string&                               tensor_decl_source,
                                const std::string&                               tensor_generate_source,
                                const std::string&                               func_call_source,
                                const std::string&                               prepare_args_source,
                                const std::string&                               make_args_source,
                                const std::string&                               fmha_flag,
                                const std::string&                               folder_name = "kernel_profile");

    std::string GenFmhaCommonKernelFunction(const std::string&                               func_name,
                                            const std::string&                               model_name,
                                            const std::unordered_map<std::string, std::any>& kernel_func_map,
                                            const std::string&                               args_decl_source,
                                            const std::string&                               func_signature_source,
                                            const std::string&                               prepare_args_source,
                                            const std::string&                               make_args_source,
                                            const std::string&                               fmha_flag,
                                            const std::string& folder_name = "kernel_profile");
};
}  // namespace flashck
