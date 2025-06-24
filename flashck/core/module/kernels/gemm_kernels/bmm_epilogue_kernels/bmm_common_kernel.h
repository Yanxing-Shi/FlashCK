#pragma once

#include <string>

#include "flashck/core/module/kernels/gemm_kernels/gemm_common_kernel.h"

/*
Common template for bmm kernels
*/

static const std::string g_bmm_extra_shape_source = R"(
{{indent}}ck::index_t stride_a = a_dim2;
{{indent}}ck::index_t stride_b = b_dim2;
{{indent}}ck::index_t stride_c = c_dim2;

{{indent}}ck::index_t batch_stride_a = a_dim1 * stride_a;
{{indent}}ck::index_t batch_stride_b = b_dim1 * stride_b;
{{indent}}ck::index_t batch_stride_c = c_dim1 * stride_c;
)";

static const std::string g_bmm_extra_header_source = R"(
{% if gemm_flag == "" %}
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
{% elif gemm_flag == "permute_m2n3" %}
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
{% elif "bias" in gemm_flag or has_d0 %}
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
{% endif %}
)";

static const std::string g_bmm_problem_args_source = R"(
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{% if "bias" in gemm_flag or gemm_flag == "add" %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% else %}
{{indent}}                                {},
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                B,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% if gemm_flag == "add" %}
{{indent}}                                std::array<ck::index_t, 1>{stride_c},
{% elif gemm_flag == "bias" %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% else %}
{{indent}}                                {},
{% endif %}
{{indent}}                                stride_c,
{{indent}}                                batch_stride_a,
{{indent}}                                batch_stride_b,
{% if gemm_flag == "add" %}
{{indent}}                                std::array<ck::index_t, 1>{batch_stride_c},
{% elif gemm_flag == "bias" %}
{{indent}}                                std::array<ck::index_t, 1>{stride_c},
{% else %}
{{indent}}                                {},
{% endif %}
{{indent}}                                batch_stride_c,
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag in ["bias", "add"] %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif gemm_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif gemm_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% endif %}
)";

static const std::string g_bmm_tensor_decl_source = R"(
    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        using namespace ck::literals;

        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, 1_uz, stride});
        }
    };

    Tensor<ADataType> a(
        f_host_tensor_descriptor(B, M, K, stride_a, batch_stride_a, ALayout{}));
    Tensor<BDataType> b(
        f_host_tensor_descriptor(B, K, N, stride_b, batch_stride_b, BLayout{}));
    Tensor<BDataType> c(
        f_host_tensor_descriptor(B, M, N, stride_c, batch_stride_c, CLayout{}));

{% if "bias" in gemm_flag %}
   Tensor<CDataType> d(f_host_tensor_descriptor(B, M, N, 0, 0, CLayout{}));
{% endif %}
{% if has_d0 %}
    Tensor<CDataType> d0(f_host_tensor_descriptor(B, M, N, stride_c, batch_stride_c, CLayout{}));
{% endif %}
{% if has_d1 %}
    Tensor<CDataType> d1(f_host_tensor_descriptor(B, M, N, stride_c, batch_stride_c, CLayout{}));
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

namespace flashck {
class BmmCommonKernel: public GemmCommonKernel {
public:
    BmmCommonKernel()          = default;
    virtual ~BmmCommonKernel() = default;

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenBmmCommonKernelProfiler(const std::string&                               model_name,
                               const std::unordered_map<std::string, std::any>& kernel_func_map,
                               const std::string&                               arg_parse,
                               const std::string&                               gemm_flag  = "",
                               const std::string&                               extra_code = "",
                               const std::string& extra_shape_template                     = g_bmm_extra_shape_source,
                               const std::string& problem_args_template                    = g_bmm_problem_args_source,
                               const std::string& extra_header_template                    = g_bmm_extra_header_source,
                               const std::string& tensor_decl_template                     = g_bmm_tensor_decl_source);

    std::string GenBmmKernelFunction(const std::string&                               func_name,
                                     const std::unordered_map<std::string, std::any>& kernel_func_map,
                                     const std::string&                               gemm_flag  = "",
                                     const std::string&                               extra_code = "",
                                     const std::string& extra_shape_template  = g_bmm_extra_shape_source,
                                     const std::string& problem_args_template = "",
                                     const std::string& extra_header_template = "");
};

}  // namespace flashck