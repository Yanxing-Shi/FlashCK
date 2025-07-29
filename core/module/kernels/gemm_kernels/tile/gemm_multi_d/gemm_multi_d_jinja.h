#pragma once

#include <string>

static const std::string g_batch_gemm_header_tpl = R"(
    #include "ck_tile/ops/gemm.hpp"
    #include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
)";


static const std::string g_gemm_make_args_tpl = R"(
    ck_tile::FlatmmHostArgs<>  args  {a_ptr,
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

static const std::string g_gemm_func_call_tpl = R"(
    {{function_name}}(
        a_m_k_dev_buf.GetDeviceBuffer(),
        b_k_n_dev_buf.GetDeviceBuffer(),
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


static const std::string g_gemm_func_signature_tpl = R"(
void {{function_name}}(
    void* a_ptr,
    void* b_ptr,
    void* ds_ptr,
    void* c_ptr,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t split_k,
    int64_t a_stride,
    int64_t b_stride,
    std::array<int64_t, DsDataType::size()> ds_stride,
    int64_t c_stride,
    hipStream_t stream
)
)";

static const std::string g_gemm_tensor_decl_tpl = R"(
    ck_tile::index_t d0_stride = ds_stride;
    ck_tile::index_t d1_stride = ds_stride;


    a_stride  = ck_tile::get_default_stride(m, k, a_stride, is_row_major(a_layout));
    b_stride  = ck_tile::get_default_stride(k, n, b_stride, is_row_major(b_layout));
    d0_stride = ck_tile::get_default_stride(m, n, d0_stride, is_row_major(d0_layout));
    d1_stride = ck_tile::get_default_stride(m, n, d1_stride, is_row_major(d1_layout));
    c_stride  = ck_tile::get_default_stride(m, n, c_stride, is_row_major(c_layout));

    ck_tile::HostTensor<ADataType> a_m_k(
        f_host_tensor_descriptor(m, k, a_stride, a_stride, a_layout));
    ck_tile::HostTensor<BDataType> b_k_n(
        f_host_tensor_descriptor(k, n, b_stride, b_stride, b_layout));
    ck_tile::HostTensor<D0DataType> d0_m_n(
        host_tensor_descriptor(M, N, StrideD0, is_row_major(d0_layout)));
    ck_tile::HostTensor<D1DataType> d1_m_n(
        host_tensor_descriptor(M, N, StrideD1, is_row_major(d1_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        f_host_tensor_descriptor(m, N, c_stride, c_stride, c_layout));

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
    ck_tile::FillUniformDistribution<D0DataType>{-5.f, 5.f}(d0_m_n);
    ck_tile::FillUniformDistribution<D1DataType>{-5.f, 5.f}(d1_m_n);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    d0_m_n_dev_buf.ToDevice(d0_m_n.data());
    d1_m_n_dev_buf.ToDevice(d1_m_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    std::array<const void*, DsDataType::size()> ds_ptr = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                              d1_m_n_dev_buf.GetDeviceBuffer()};

    std::array<ck_tile::index_t, DsDataType::size()> ds_stride = {d0_stride, d1_stride};
)";

static const std::string g_gemm_create_args_tpl = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("split_k", "1", "Split k value")
        .insert("a_stride", "0", "Tensor A stride")
        .insert("b_stride", "0", "Tensor B stride")
        .insert("ds_stride", "0", "Tensor Ds stride")
        .insert("c_stride", "0", "Tensor C stride");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_group_gemm_arg_parser_tpl = R"(

    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    ck_tile::index_t k = arg_parser.get_int("k");
    ck_tile::index_t split_k         = arg_parser.get_int("split_k");


    ck_tile::index_t a_stride = arg_parser.get_int("a_stride");
    ck_tile::index_t b_stride = arg_parser.get_int("b_stride");
    ck_tile::index_t ds_stride = arg_parser.get_int("ds_stride");
    ck_tile::index_t c_stride = arg_parser.get_int("c_stride");

)";
