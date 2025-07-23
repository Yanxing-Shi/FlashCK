#pragma once

#include "core/module/kernels/kernel.h"
#include "core/module/kernels/kernel_registry.h"
#include "core/module/kernels/norm_kernels/norm_common_kernel.h"

/// @brief RMSNorm type configuration template for different data types
static const std::string g_rms_norm_dtype_config_utils_tpl = R"(

template <typename InType,
          typename OutType,
          typename SmoothScaleDataType_,
          typename YScaleDataType_>
struct RMSNormTypeConfig;

template <typename OutType, typename SmoothScaleDataType_, typename YScaleDataType_>
struct RMSNormTypeConfig<ck_tile::fp32_t, OutType, SmoothScaleDataType_, YScaleDataType_>
{
    using XDataType           = ck_tile::fp32_t;
    using YDataType           = OutType;
    using GammaDataType       = ck_tile::fp32_t;
    using InvRmsDataType      = ck_tile::fp32_t;
    using UnquantYDataType    = ck_tile::null_type;
    using SmoothScaleDataType = SmoothScaleDataType_;
    using YScaleDataType      = YScaleDataType_;
    using ComputeDataType     = ck_tile::fp32_t;
};

template <typename OutType, typename SmoothScaleDataType_, typename YScaleDataType_>
struct RMSNormTypeConfig<ck_tile::half_t, OutType, SmoothScaleDataType_, YScaleDataType_>
{
    using XDataType           = ck_tile::half_t;
    using YDataType           = OutType;
    using GammaDataType       = ck_tile::half_t;
    using InvRmsDataType      = ck_tile::half_t;
    using UnquantYDataType    = ck_tile::null_type;
    using SmoothScaleDataType = SmoothScaleDataType_;
    using YScaleDataType      = YScaleDataType_;
    using ComputeDataType     = ck_tile::fp32_t;
};

template <typename OutType, typename SmoothScaleDataType_, typename YScaleDataType_>
struct RMSNormTypeConfig<ck_tile::bf16_t, OutType, SmoothScaleDataType_, YScaleDataType_>
{
    using XDataType           = ck_tile::bf16_t;
    using YDataType           = OutType;
    using GammaDataType       = ck_tile::bf16_t;
    using InvRmsDataType      = ck_tile::bf16_t;
    using UnquantYDataType    = ck_tile::null_type;
    using SmoothScaleDataType = SmoothScaleDataType_;
    using YScaleDataType      = YScaleDataType_;
    using ComputeDataType     = ck_tile::fp32_t;
};

)";

/// @brief RMSNorm data type declaration template
static const std::string g_rms_norm_dtype_decl_tpl = R"(
using TypeConfig = RMSNormTypeConfig<{{x_dtype}}, {{y_dtype}}, {{smooth_scale_dtype}}, {{y_scale_dtype}}>;

using XDataType         = typename TypeConfig::XDataType;
using YDataType         = typename TypeConfig::YDataType;
using GammaDataType     = typename TypeConfig::GammaDataType;
using XResidualDataType = XDataType;
using YResidualDataType = XDataType;

using InvRmsDataType = typename TypeConfig::InvRmsDataType;
using UnquantYDataType = typename TypeConfig::UnquantYDataType;

using SmoothScaleDataType = typename TypeConfig::SmoothScaleDataType;
using YScaleDataType = typename TypeConfig::YScaleDataType;

using ComputeDataType = typename TypeConfig::ComputeDataType;
)";

/// @brief RMSNorm argument creation template with conditional compilation
static const std::string g_rms_norm_make_args_tpl = R"(

    ck_tile::Rmsnorm2dFwdHostArgs args{x_ptr,
{% if fused_add == "pre_add" or fused_add == "pre_add_store" %}
                              x_residual_ptr,
{% else %}                              
                              nullptr,
{% endif %}
{% if fused_quant == "smooth_dynamic_quant" %}
                              smooth_scale_ptr,
{% else %}
                              nullptr,
{% endif %}
                              gamma_ptr,
                              y_ptr,
{% if fused_add == "pre_add_store" %}
                              y_residual_ptr,
{% else %}                                
                              nullptr,
{% endif %}
{% if fused_quant == "dynamic_quant" or fused_quant == "smooth_dynamic_quant" %}
                              y_scale_ptr,
{% else %}
                              nullptr,
{% endif %}
                              nullptr, // p_invRms
                              nullptr, // p_y_unquant
                              eps,
                              static_cast<ck_tile::index_t>(m),
                              static_cast<ck_tile::index_t>(n),
                              static_cast<ck_tile::index_t>(x_stride),
                              static_cast<ck_tile::index_t>(xr_stride),
                              static_cast<ck_tile::index_t>(y_stride),
                              static_cast<ck_tile::index_t>(yr_stride)};

)";

/// @brief RMSNorm function call template with conditional buffer usage
static const std::string g_rms_norm_func_call_tpl = R"(
    {{function_name}}(
        x_buf.GetDeviceBuffer(),
{% if fused_add == "pre_add" or fused_add == "pre_add_store" %}
        x_residual_buf.GetDeviceBuffer(),
{% else %} 
        nullptr,
{% endif %} 
{% if fused_quant == "smooth_dynamic_quant" %}
        sm_scale_buf.GetDeviceBuffer(),
{% else %} 
        nullptr,
{% endif %}         
        gamma_buf.GetDeviceBuffer(),
        y_buf.GetDeviceBuffer(),
{% if fused_add == "pre_add_store" %}
        y_residual_buf.GetDeviceBuffer(),
{% else %} 
        nullptr,
{% endif %}
{% if fused_quant == "dynamic_quant" or fused_quant == "smooth_dynamic_quant" %}
        y_scale_buf.GetDeviceBuffer(),
{% else %} 
        nullptr,
{% endif %}
        m,
        n,
        epsilon,
        x_stride,
        xr_stride,
        y_stride,
        yr_stride,
        stream
    );
)";

/// @brief RMSNorm function signature template
static const std::string g_rms_norm_func_signature_tpl = R"(
void {{function_name}}(
    void* x_ptr,
    void* x_residual_ptr,
    void* smooth_scale_ptr,
    void* gamma_ptr,
    void* y_ptr,
    void* y_residual_ptr,  
    void* y_scale_ptr,
    int64_t m,
    int64_t n,
    float eps,
    int64_t x_stride,
    int64_t xr_stride,
    int64_t y_stride,
    int64_t yr_stride,
    hipStream_t stream
)
)";

/// @brief RMSNorm tensor declaration template for profiling
static const std::string g_rms_norm_tensor_decl_tpl = R"(
    // host verify
    ck_tile::HostTensor<XDataType>     x_host({m, n}, {x_stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});

{% if fused_add == "pre_add" or fused_add == "pre_add_store" %}
    ck_tile::HostTensor<XResidualDataType> x_residual_host({m, n}, {xr_stride, 1});
{% endif %}

{% if fused_add == "pre_add_store" %}
    ck_tile::HostTensor<YResidualDataType> y_residual_host({m, n}, {yr_stride, 1});
{% endif %}

    ck_tile::HostTensor<YDataType> y_host_dev({m, n}, {y_stride, 1});

{% if fused_quant == "dynamic_quant" or fused_quant == "smooth_dynamic_quant" %}
    ck_tile::HostTensor<YScaleDataType>          y_scale_host_dev({m});
{% endif %}

{% if fused_quant == "smooth_dynamic_quant" %}
    ck_tile::HostTensor<SmoothScaleDataType> sm_scale_host_dev({n});
{% endif %}

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
{% if fused_add == "pre_add" or fused_add == "pre_add_store" %}
    ck_tile::FillUniformDistribution<XResidualDataType>{-.5f, .5f}(x_residual_host);
{% endif %}
{% if fused_quant == "smooth_dynamic_quant" %}
    ck_tile::FillUniformDistribution<SmoothScaleDataType>{-1.f, 1.f}(sm_scale_host_dev);
{% endif %}
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
{% if fused_quant == "dynamic_quant" or fused_quant == "smooth_dynamic_quant" %}
    ck_tile::DeviceMem y_scale_buf(y_scale_host_dev.get_element_space_size_in_bytes());
{% endif %}
{% if fused_quant == "smooth_dynamic_quant" %}
    ck_tile::DeviceMem sm_scale_buf(sm_scale_host_dev.get_element_space_size_in_bytes());
{% endif %}

{% if fused_add == "pre_add" or fused_add == "pre_add_store" %}
    ck_tile::DeviceMem x_residual_buf(x_residual_host.get_element_space_size_in_bytes());
{% endif %}

{% if fused_add == "pre_add_store" %}
    ck_tile::DeviceMem y_residual_buf(y_residual_host.get_element_space_size_in_bytes());
{% endif %}

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());
{% if fused_add == "pre_add" or fused_add == "pre_add_store" %}
    x_residual_buf.ToDevice(x_residual_host.data());
{% endif %}
{% if fused_quant == "smooth_dynamic_quant" %}
    sm_scale_buf.ToDevice(sm_scale_host.data());  
{% endif %}


)";

namespace flashck {

/**
 * @brief RMSNorm kernel implementation
 *
 * Implements Root Mean Square Normalization operation with support for:
 * - Multiple data types (FP16, FP32, BF16)
 * - Residual connections
 * - Quantization support
 * - No bias parameter (unlike LayerNorm)
 */
class RMSNormKernel: public NormCommonKernel {
public:
    /// @brief Generate tuning code for RMSNorm kernel
    /// @param model_name Name of the model being tuned
    /// @param kind_name Kind/type identifier ("rms_norm")
    /// @param instance_map Map of kernel instances and configurations
    /// @param folder_name Output folder for generated code
    /// @return Vector of tuples containing source and object file paths
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const std::string&    model_name,
                     const std::string&    kind_name,
                     const instance_map_t& instance_map,
                     const std::string&    folder_name = "kernel_profile") override;

    /// @brief Generate runtime code for RMSNorm kernel
    /// @param func_name Function name for the generated kernel
    /// @param model_name Name of the model
    /// @param running_infos Runtime configuration information
    /// @param instance_map Map of kernel instances and configurations
    /// @param folder_name Output folder for generated code
    /// @return Generated source code as string
    std::string CodeGenForRunning(const std::string&                        func_name,
                                  const std::string&                        model_name,
                                  const std::map<std::string, RunningItem>& running_infos,
                                  const instance_map_t&                     instance_map,
                                  const std::string&                        folder_name = "kernel_profile") override;

    /// @brief Execute RMSNorm kernel with given arguments
    /// @param kernel_func_name Name of the kernel function to launch
    /// @param args Kernel arguments containing tensors and parameters
    void KernelLauncher(const std::string& kernel_func_name, const KernelArgs_t& args) override;
};

}  // namespace flashck

/// @brief Register RMSNorm kernel for TILE source with FP16, FP32, BF16 support
FC_REGISTER_KERNEL(TILE, rms_norm, flashck::RMSNormKernel, ALL_LAYOUT, FP16, FP32, BF16);
