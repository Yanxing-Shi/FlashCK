#include "lightinfer/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_bias_permute_m2n3_kernel.h"

#include "lightinfer/core/profiler/library.h"

static const std::string g_args_parser_source_m2n3 = R"(
  ck::index_t K = std::stoi(argv[1]);
  ck::index_t M = std::stoi(argv[2]);
  ck::index_t N = std::stoi(argv[3]);
  ck::index_t G1 = std::atoi(argv[4]);
  ck::index_t G2 = std::atoi(argv[5]);
  ck::index_t G3 = std::atoi(argv[6]);
  ck::index_t split_k = std::atoi(argv[7]);
  ck::index_t a_dim0 = M;
  ck::index_t a_dim1 = K;
  ck::index_t b_dim0 = N;
  ck::index_t b_dim1 = K;
  ck::index_t c_dim0 = M;
  ck::index_t c_dim1 = N;
  ck::index_t p_dim0 = G1;
  ck::index_t p_dim1 = G2;
  ck::index_t p_dim2 = G3;
)";

static const std::string g_inverse_shape_source_m2n3 = R"(
    ck::index_t G1 = p_dim0; // G1
    ck::index_t G2 = p_dim1; // G2
    ck::index_t G3 = p_dim2; // G3
)";

static const std::string g_extra_shape_source_m2n3 = R"(
    ck::index_t M0 = M / G1;
    ck::index_t M1 = G1;
    ck::index_t N0 = G2;
    ck::index_t N1 = G3;
    ck::index_t N2 = N / G2 / G3;

    ck::index_t K0 = K;
    ck::index_t G = 1;

    // A[G, M0, M1, K0]
    std::vector<ck::index_t> a_ms_ks_lengths{G, M0, M1, K0};
    std::vector<ck::index_t> a_ms_ks_strides{M0*M1*K0, M1 * K0, K0, 1};

    // B[G, N0, N1, N2, K0]
    std::vector<ck::index_t> b_ns_ks_lengths{G, N0, N1, N2, K0};
    std::vector<ck::index_t> b_ns_ks_strides{N0*N1*N2*K0, N1 * N2 * K0, N2 * K0, K0, 1};

    // D[G, N0, M0, N1, M1, N2]
    std::vector<ck::index_t> d_ms_ns_lengths{G, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> d_ms_ns_strides{N0 * N1 * N2, 0, 0, N1 * N2, N2, 1};

    // E[G, N0, M0, N1, M1, N2] 2, 0, 3, 1, 4
    // std::vector<ck::index_t> e_ms_ns_lengths{G, M0, M1, N0, N1, N2};
    // std::vector<ck::index_t> e_ms_ns_strides{M0* M1* N0* N1* N2,
    //                                            N1 * M1 * N2,
    //                                            N2,
    //                                            M0 * N1 * M1 * N2,
    //                                            M1 * N2,
    //                                            1};

    // E[G, N0, M0, M1, N1, N2] 2, 0, 1, 3, 4
    std::vector<ck::index_t> e_ms_ns_lengths{G, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> e_ms_ns_strides{M0* M1* N0* N1* N2,
                                               N1 * M1 * N2,
                                               M1 * N2,
                                               M0 * N1 * M1 * N2,
                                               N2,
                                               1};
)";

static const std::string g_tensor_decl_source_m2n3 = R"(
    Tensor<ADataType> a_ms_ks(a_ms_ks_lengths, a_ms_ks_strides);
    Tensor<BDataType> b_ns_ks(b_ns_ks_lengths, b_ns_ks_strides);
    Tensor<BDataType> d_ms_ns(d_ms_ns_lengths, d_ms_ns_strides);
    Tensor<CDataType> e_ms_ns(e_ms_ns_lengths, e_ms_ns_strides);

    a_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    b_ns_ks.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    d_ms_ns.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});

    DeviceMem a_device_buf(sizeof(ADataType) * a_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(BDataType) * d_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(CDataType) *
                           e_ms_ns.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_ms_ks.mData.data());
    b_device_buf.ToDevice(b_ns_ks.mData.data());
    d_device_buf.ToDevice(d_ms_ns.mData.data());
    e_device_buf.SetZero();

    auto in_dev_buff_ptr = a_device_buf.GetDeviceBuffer();
    auto weight_dev_buff_ptr = b_device_buf.GetDeviceBuffer();
    auto bias_dev_buff_ptr = d_device_buf.GetDeviceBuffer();
    auto out_dev_buff_ptr = e_device_buf.GetDeviceBuffer();
)";

namespace lightinfer {

std::map<std::string, std::shared_ptr<void>> GemmRCRBiasPermuteM2N3Kernel::Init(const OperationKind&   op_kind,
                                                                                const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<GemmOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmRCRBiasPermuteM2N3Kernel::GenKernelProfiler(const std::string&                               model_name,
                                                const std::unordered_map<std::string, std::any>& kernel_func_map,
                                                const std::string&                               folder_name)
{

    std::string args_parser = TemplateLoadAndRender(g_args_parser_source_m2n3, {{}});

    std::string inverse_shape = TemplateLoadAndRender(g_inverse_shape_source_m2n3, {{}});

    return GenGemmCommonKernelProfiler(model_name,
                                       kernel_func_map,
                                       args_parser,
                                       "bias_permute_m2n3",
                                       "",
                                       2,
                                       g_extra_shape_source_m2n3,
                                       g_problem_args_source,
                                       g_extra_header_source,
                                       g_tensor_decl_source_m2n3,
                                       inverse_shape);
}

std::string
GemmRCRBiasPermuteM2N3Kernel::GenKernelFunction(const std::string&                               func_name,
                                                const std::string&                               model_name,
                                                const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    std::string inverse_shape = TemplateLoadAndRender(g_inverse_shape_source_m2n3, {{}});

    return GenGemmCommonKernelFunction(func_name,
                                       kernel_func_map,
                                       "bias_permute_m2n3",
                                       "",
                                       2,
                                       g_extra_shape_source_m2n3,
                                       g_problem_args_source,
                                       g_extra_header_source,
                                       inverse_shape);
}

void GemmRCRBiasPermuteM2N3Kernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{

    auto gemm_args = std::get<GemmKernelArgs>(args);

    VLOG(1) << gemm_args.GetDimInfo();

    decltype(&GemmBiasPermute) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    kernel_func(gemm_args.in_ptr_,
                gemm_args.weight_ptr_,
                gemm_args.out_ptr_,
                gemm_args.bias_ptr_,
                gemm_args.a_dim0_,
                gemm_args.a_dim1_,
                gemm_args.b_dim0_,
                gemm_args.b_dim1_,
                gemm_args.c_dim0_,
                gemm_args.c_dim1_,
                gemm_args.p_dim0_,
                gemm_args.p_dim1_,
                gemm_args.p_dim2_,
                gemm_args.stream_);
}
}  // namespace lightinfer