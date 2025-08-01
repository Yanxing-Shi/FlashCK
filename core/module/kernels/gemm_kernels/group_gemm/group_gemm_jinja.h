#pragma once

#include <string>

static const std::string g_group_gemm_header_tpl = R"(
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
)";

static const std::string g_group_gemm_create_args_tpl = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("split_k", "1", "Split k value")
        .insert("group_count", "1", "Group count")
        .insert("a_stride", "0", "Tensor A stride")
        .insert("b_stride", "0", "Tensor B stride")
        .insert("c_stride", "0", "Tensor C stride");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_group_gemm_arg_parser_tpl = R"(
    std::vector<ck_tile::index_t> m = arg_parser.get_int_vec("m");
    std::vector<ck_tile::index_t> n = arg_parser.get_int_vec("n");
    std::vector<ck_tile::index_t> k = arg_parser.get_int_vec("k");
    ck_tile::index_t split_k = arg_parser.get_int("split_k");
    ck_tile::index_t group_count = arg_parser.get_int("group_count");

    std::vector<ck_tile::index_t> a_stride = arg_parser.get_int_vec("a_stride");
    std::vector<ck_tile::index_t> b_stride = arg_parser.get_int_vec("b_stride");
    std::vector<ck_tile::index_t> c_stride = arg_parser.get_int_vec("c_stride");

    if(!valid_input_data(group_count, m, n, k, a_stride, b_stride, c_stride))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;
        for(int i = 0; i < group_count; i++)
        {
            m.push_back(256 + 256 * i);
            n.push_back(256 + 512 * i);
            k.push_back(512 + 128 * i);

            a_stride.push_back(k[i]);
            b_stride.push_back(k[i]);
            c_stride.push_back(n[i]);
        }
    }

)";

static const std::string g_group_gemm_make_args_tpl = R"(
    ck_tile::GroupedGemmHostArgs  args{a_ptr,
                                      b_ptr,
                                      c_ptr,
                                      split_k,
                                      m,
                                      n,
                                      k,
                                      a_stride,
                                      b_stride,
                                      c_stride};
)";

static const std::string g_group_gemm_func_call_tpl = R"(
    {{function_name}}(
        a_m_k_dev_buf.GetDeviceBuffer(),
        b_k_n_dev_buf.GetDeviceBuffer(),
        c_m_n_dev_buf.GetDeviceBuffer(),
        m,
        n,
        k,
        split_k,
        a_stride,
        b_stride,
        c_stride,
        stream
    );
)";


static const std::string g_group_gemm_func_signature_tpl = R"(
void {{function_name}}(
    void* a_ptr,
    void* b_ptr,
    void* c_ptr,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t split_k,
    int64_t a_stride,
    int64_t b_stride,
    int64_t c_stride,
    hipStream_t stream
)
)";

static const std::string g_group_gemm_tensor_decl_tpl = R"(
    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

    a_m_k_tensors.reserve(group_count);
    b_k_n_tensors.reserve(group_count);
    c_m_n_tensors.reserve(group_count);

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

    a_m_k_dev_buf.reserve(group_count);
    b_k_n_dev_buf.reserve(group_count);
    c_m_n_dev_buf.reserve(group_count);

    std::vector<grouped_gemm_kargs> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        stride_As[i] = ck_tile::get_default_stride(M, N, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(b_layout))));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));

        std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                  << " b_k_n: " << b_k_n_tensors[i].mDesc << " c_m_n: " << c_m_n_tensors[i].mDesc
                  << std::endl;

        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);

        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            a_m_k_tensors[i].get_element_space_size_in_bytes()));
        b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            b_k_n_tensors[i].get_element_space_size_in_bytes()));
        c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            c_m_n_tensors[i].get_element_space_size_in_bytes()));

        a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
        b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
        c_m_n_dev_buf[i]->SetZero();
        c_m_n_tensors[i].SetZero();

        const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
        const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
        void* p_c       = c_m_n_dev_buf[i]->GetDeviceBuffer();

        gemm_descs.push_back(
            {p_a, p_b, p_c, kbatch, M, N, K, stride_As[i], stride_Bs[i], stride_Cs[i]});
    }

)";

