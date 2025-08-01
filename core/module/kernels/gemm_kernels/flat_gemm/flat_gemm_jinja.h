#pragma once

#include <string>

static const std::string g_flat_gemm_header_tpl = R"(
#include "ck_tile/ops/flatmm.hpp"
)";

static const std::string g_flat_gemm_create_args_tpl = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("split_k", "1", "Split k value")
        .insert("a_stride", "0", "Tensor A stride")
        .insert("b_stride", "0", "Tensor B stride")
        .insert("ds_stride", "0", "Tensor DS stride")
        .insert("c_stride", "0", "Tensor C stride");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_flat_gemm_make_args_tpl = R"(
    ck_tile::FlatmmHostArgs<{{tensor_num_d}}>  args {a_ptr,
                                                    b_ptr,
                                                    ds_ptr,
                                                    c_ptr,
                                                    split_k,
                                                    m,
                                                    n,
                                                    k,
                                                    a_stride,
                                                    b_stride,
                                                    ds_stride,
                                                    c_stride};
)";

static const std::string g_flat_gemm_arg_parser_tpl = R"(

    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    ck_tile::index_t k = arg_parser.get_int("k");
    ck_tile::index_t split_k = arg_parser.get_int("split_k");

    ck_tile::index_t a_stride = arg_parser.get_int("a_stride");
    ck_tile::index_t b_stride = arg_parser.get_int("b_stride");
    ck_tile::index_t ds_stride = arg_parser.get_int("ds_stride");
    ck_tile::index_t c_stride = arg_parser.get_int("c_stride");

)";


static const std::string g_flat_gemm_func_call_tpl = R"(
    {{function_name}}(
        a_m_k_dev_buf.GetDeviceBuffer(),
        b_shuffle_dev_buf.GetDeviceBuffer(),
        ds_ptr,
        c_m_n_dev_buf.GetDeviceBuffer(),
        m,
        n,
        k,
        split_k,
        a_stride,
        b_stride,
        ds_stride,
        c_stride,
        stream
    );
)";


static const std::string g_flat_gemm_func_signature_tpl = R"(
void {{function_name}}(
    void* a_ptr,
    void* b_ptr,
    std::array<void*, {{tensor_num_d}}> ds_ptr,
    void* c_ptr,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t split_k,
    int64_t a_stride,
    int64_t b_stride,
    std::array<int64_t, {{tensor_num_d}}> ds_stride,
    int64_t c_stride,
    hipStream_t stream
)
)";

static const std::string g_flat_gemm_tensor_decl_tpl = R"(

    {% for i in range(tensor_num_d) %}
    ck_tile::index_t d{{i}}_stride = ds_stride;
    {% endfor %}

    a_stride  = ck_tile::get_default_stride(m, k, a_stride, is_row_major(a_layout));
    b_stride  = ck_tile::get_default_stride(k, n, b_stride, is_row_major(b_layout));
    {% for i in range(tensor_num_d) %}
    d{{i}}_stride = ck_tile::get_default_stride(m, n, d{{i}}_stride, is_row_major(d{{i}}_layout));
    {% endfor %}
    c_stride  = ck_tile::get_default_stride(m, n, c_stride, is_row_major(c_layout));

    ck_tile::HostTensor<ADataType> a_m_k(
        f_host_tensor_descriptor(m, k, a_stride, a_stride, a_layout));
    ck_tile::HostTensor<BDataType> b_k_n(
        f_host_tensor_descriptor(k, n, b_stride, b_stride, b_layout));
    {% for i in range(tensor_num_d) %}
    ck_tile::HostTensor<D{{i}}DataType> d{{i}}_m_n(
        host_tensor_descriptor(M, N, d{{i}}_stride, is_row_major(d{{i}}_layout)));
    {% endfor %}
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        f_host_tensor_descriptor(m, N, c_stride, c_stride, c_layout));

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
    {% for i in range(tensor_num_d) %}
    ck_tile::FillUniformDistribution<D{{i}}DataType>{-5.f, 5.f}(d{{i}}_m_n);
    {% endfor %}

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    {% for i in range(tensor_num_d) %}
    ck_tile::DeviceMem d{{i}}_m_n_dev_buf(d{{i}}_m_n.get_element_space_size_in_bytes());
    {% endfor %}
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    {% for i in range(tensor_num_d) %}
    d{{i}}_m_n_dev_buf.ToDevice(d{{i}}_m_n.data());
    {% endfor %}
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    std::array<const void*, DsDataType::size()> ds_ptr = {
        {% for i in range(tensor_num_d) %}d{{i}}_m_n_dev_buf.GetDeviceBuffer(){% if not loop.last %}, {% endif %}{% endfor %}
    };

    std::array<ck_tile::index_t, DsDataType::size()> ds_stride = {
        {% for i in range(tensor_num_d) %}d{{i}}_stride{% if not loop.last %}, {% endif %}{% endfor %}
    };

    ck_tile::HostTensor<BDataType> b_shuffle_host = shuffle_b(b_k_n);
    ck_tile::DeviceMem b_shuffle_dev_buf(b_shuffle_host.get_element_space_size_in_bytes());
    b_shuffle_dev_buf.ToDevice(b_shuffle_host.data());
)";

