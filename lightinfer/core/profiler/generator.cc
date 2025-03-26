#include "lightinfer/core/profiler/generator.h"

#include <numeric>

#include "lightinfer/core/profiler/library.h"
#include "lightinfer/core/utils/enforce.h"

namespace lightinfer {

template<typename F>
void CreateXlopsGemmKernel(F f, const LayoutType& a_layout, const LayoutType& b_layout, bool is_split_k = false)
{

    std::vector<GemmTileDesc> tile_descriptions;

    if (is_split_k) {
        tile_descriptions = {
            // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| 
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| 
//       |      |      |      |    |    |     |     | Wave| Wave|
//       |      |      |      |    |    |     |     |     |     |
  {   256,   256,   128,    32,   8,   8,   32,   32,    4,    2},
            // clang-format on
        };
    }
    else {
        tile_descriptions = {
            // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| 
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| 
//       |      |      |      |    |    |     |     | Wave| Wave|
//       |      |      |      |    |    |     |     |     |     |
  {   256,   256,   128,    32,   8,   8,   32,   32,    4,    2},
  {   256,   128,   256,    32,   8,   8,   32,   32,    2,    4},
  {   128,   128,   128,    32,   8,   8,   32,   32,    4,    2},
  {   256,   128,   128,    32,   8,   8,   32,   32,    2,    2},
  {   128,   128,    64,    32,   8,   8,   32,   32,    2,    2},
  {   128,    64,   128,    32,   8,   8,   32,   32,    2,    2},
  {   256,   128,    64,    32,   8,   8,   32,   32,    2,    1},
  {   256,    64,   128,    32,   8,   8,   32,   32,    1,    2},
            // clang-format on
        };
    }

    std::vector<BlockTransferDesc> a_block_descriptions_rowmajor;
    if (is_split_k) {
        a_block_descriptions_rowmajor = {
            // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {{1, 4, 64, 1},      {0, 2, 1, 3},    {0, 2, 1, 3},             3,              8,              8,         1},
            // clang-format on
        };
    }
    else {
        a_block_descriptions_rowmajor = {
            // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
  {     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
  {     {4, 32, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
  {     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
  {     {4, 32, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
  {     {4, 32, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
  {     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
  {     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
            // clang-format on
        };
    }

    //     std::vector<BlockTransferDesc> a_block_descriptions_colmajor = {
    //         // clang-format off
    // //  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
    // //   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
    // // Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
    // //                |               |               |               |               |               |          |
    //         // clang-format on
    //         {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 8, 1},
    //         {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
    //         {{4, 32, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 8, 1},
    //         {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
    //         {{4, 32, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 8, 1},
    //         {{4, 32, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
    //         {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
    //         {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 1, 8, 1},
    //     };

    //     std::vector<BlockTransferDesc> b_block_descriptions_rowmajor = {
    //         // clang-format off
    // //  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
    // //   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
    // // Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
    // //                |               |               |               |               |               |          |
    //   {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
    //   {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              4,              8,         1},
    //   {    {4, 32, 1},     {0, 2, 1},     {0, 2, 1},              1,              4,              8,         1},
    //   {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
    //   {    {4, 32, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
    //   {    {4, 32, 1},     {0, 2, 1},     {0, 2, 1},              1,              4,              8,         1},
    //   {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              1,              8,         1},
    //   {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
    //         // clang-format on
    //     };

    std::vector<BlockTransferDesc> b_block_descriptions_colmajor;
    if (is_split_k) {
        b_block_descriptions_colmajor = {
            // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {{1, 4, 64, 1},      {0, 1, 3, 2},    {0, 1, 3, 2},             3,              8,              8,         1},
            // clang-format on
        };
    }
    else {
        b_block_descriptions_colmajor = {
            // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
  {    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
  {    {4, 32, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
  {    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
  {    {4, 32, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
  {    {4, 32, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
  {    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
  {    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
            // clang-format on
        };
    }

    std::vector<CBlockTransferDesc> c_block_descriptions = {
        // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
  {1, 1,              {1, 32, 1, 8},               8},
  {1, 1,              {1, 32, 1, 8},               8},
  {1, 1,              {1, 16, 1, 8},               8},
  {1, 1,              {1, 32, 1, 8},               8},
  {1, 1,              {1, 32, 1, 4},               8},
  {1, 1,              {1, 16, 1, 8},               8},
  {1, 1,              {1, 32, 1, 8},               8},
  {1, 1,              {1, 32, 1, 8},               8},
        // clang-format on
    };

    // const auto a_block_descriptions =
    //     (a_layout == LayoutType::RowMajor) ? a_block_descriptions_rowmajor : a_block_descriptions_colmajor;
    // const auto b_block_descriptions =
    //     (b_layout == LayoutType::RowMajor) ? b_block_descriptions_rowmajor : b_block_descriptions_colmajor;

    const auto a_block_descriptions = a_block_descriptions_rowmajor;
    const auto b_block_descriptions = b_block_descriptions_colmajor;

    // check size
    LI_ENFORCE_EQ(
        tile_descriptions.size(),
        a_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), a_block_descriptions.size()));
    LI_ENFORCE_EQ(
        tile_descriptions.size(),
        b_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), b_block_descriptions.size()));
    LI_ENFORCE_EQ(
        tile_descriptions.size(),
        c_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), c_block_descriptions.size()));

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < tile_descriptions.size(); i++) {
        std::shared_ptr<GemmOperation> gemm_operation = std::make_shared<GemmOperation>();
        gemm_operation->tile_desc_                    = tile_descriptions[i];
        gemm_operation->a_block_transfer_             = a_block_descriptions[i];
        gemm_operation->b_block_transfer_             = b_block_descriptions[i];
        gemm_operation->c_block_transfer_             = c_block_descriptions[i];

        auto all = f(gemm_operation);
        emitters_ptr->Append(std::move(all));
    }
}

// Gemm/Bmm + any epilogue
void CreateGemmOperations(const GemmProblem& problem)
{

    auto a_layout = std::get<0>(problem.GetLayout());
    auto b_layout = std::get<1>(problem.GetLayout());
    auto c_layout = std::get<2>(problem.GetLayout());

    bool is_split_k = problem.operation_kind_ == GemmOperationKind::SplitKGemm;

    // Gemm
    if (problem.epilogue_op_ == TensorOperation::PassThrough) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = problem.operation_kind_;

                if (problem.operation_kind_ == GemmOperationKind::Gemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmXdl_CShuffle;
                }
                else if (problem.operation_kind_ == GemmOperationKind::BatchGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceBatchedGemmMultiD_Xdl;
                }
                else if (problem.operation_kind_ == GemmOperationKind::SplitKGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmXdlSplitKCShuffle;
                }
                else {
                    LI_THROW(Unimplemented("unsuuport gemm operation kind"));
                }

                gemm_op->a_tensor_desc_ = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_ = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_ = TensorDesc(problem.c_dtype_, c_layout);

                gemm_op->accumulator_type_ = problem.acc_dtype_;
                gemm_op->e_dtype_          = problem.c_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::PassThrough;

                gemm_op->epilogue_op_ = TensorOperation::PassThrough;

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout,
            is_split_k);
    }

    // Gemm + bias
    else if (problem.epilogue_op_ == TensorOperation::Add) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = problem.operation_kind_;

                if (problem.operation_kind_ == GemmOperationKind::Gemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmMultipleD_Xdl_CShuffle;
                }
                else if (problem.operation_kind_ == GemmOperationKind::BatchGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceBatchedGemmMultiD_Xdl;
                }
                else if (problem.operation_kind_ == GemmOperationKind::SplitKGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemm_Xdl_CShuffleV3R1;
                }
                else {
                    LI_THROW(Unimplemented("unsuuport gemm operation kind"));
                }

                gemm_op->a_tensor_desc_ = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_ = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_ = TensorDesc(problem.c_dtype_, c_layout);

                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::Add;

                gemm_op->epilogue_op_ = TensorOperation::Add;

                gemm_op->ds_layout_ = {c_layout};
                gemm_op->ds_dtype_  = {problem.c_dtype_};

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
    // Gemm + bias + add
    else if (problem.epilogue_op_ == TensorOperation::AddAdd) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = problem.operation_kind_;
                if (problem.operation_kind_ == GemmOperationKind::Gemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmMultipleD_Xdl_CShuffle;
                }
                else if (problem.operation_kind_ == GemmOperationKind::BatchGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceBatchedGemmMultiD_Xdl;
                }
                else if (problem.operation_kind_ == GemmOperationKind::SplitKGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemm_Xdl_CShuffleV3R1;
                }
                else {
                    LI_THROW(Unimplemented("unsuuport gemm operation kind"));
                }

                gemm_op->a_tensor_desc_ = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_ = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_ = TensorDesc(problem.c_dtype_, c_layout);

                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::AddAdd;

                gemm_op->epilogue_op_ = TensorOperation::AddAdd;

                gemm_op->ds_layout_ = {c_layout, c_layout};
                gemm_op->ds_dtype_  = {problem.c_dtype_, problem.c_dtype_};

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
    else if (problem.epilogue_op_ == TensorOperation::AddSiLU) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = problem.operation_kind_;
                if (problem.operation_kind_ == GemmOperationKind::Gemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmMultipleD_Xdl_CShuffle;
                }
                else if (problem.operation_kind_ == GemmOperationKind::BatchGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceBatchedGemmMultiD_Xdl;
                }
                else if (problem.operation_kind_ == GemmOperationKind::SplitKGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemm_Xdl_CShuffleV3R1;
                }
                else {
                    LI_THROW(Unimplemented("unsuuport gemm operation kind"));
                }

                gemm_op->a_tensor_desc_    = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_    = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_    = TensorDesc(problem.c_dtype_, c_layout);
                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::AddSiLU;

                gemm_op->epilogue_op_ = TensorOperation::AddSiLU;

                gemm_op->ds_layout_ = {c_layout};
                gemm_op->ds_dtype_  = {problem.c_dtype_};

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
    else if (problem.epilogue_op_ == TensorOperation::AddTanh) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = problem.operation_kind_;
                if (problem.operation_kind_ == GemmOperationKind::Gemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmMultipleD_Xdl_CShuffle;
                }
                else if (problem.operation_kind_ == GemmOperationKind::BatchGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceBatchedGemmMultiD_Xdl;
                }
                else if (problem.operation_kind_ == GemmOperationKind::SplitKGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemm_Xdl_CShuffleV3R1;
                }
                else {
                    LI_THROW(Unimplemented("unsuuport gemm operation kind"));
                }

                gemm_op->a_tensor_desc_    = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_    = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_    = TensorDesc(problem.c_dtype_, c_layout);
                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::AddTanh;

                gemm_op->epilogue_op_ = TensorOperation::AddTanh;

                gemm_op->ds_layout_ = {c_layout};
                gemm_op->ds_dtype_  = {problem.c_dtype_};

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
    else if (problem.epilogue_op_ == TensorOperation::AddGelu) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = problem.operation_kind_;
                if (problem.operation_kind_ == GemmOperationKind::Gemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmMultipleD_Xdl_CShuffle;
                }
                else if (problem.operation_kind_ == GemmOperationKind::BatchGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceBatchedGemmMultiD_Xdl;
                }
                else if (problem.operation_kind_ == GemmOperationKind::SplitKGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemm_Xdl_CShuffleV3R1;
                }
                else {
                    LI_THROW(Unimplemented("unsuuport gemm operation kind"));
                }

                gemm_op->a_tensor_desc_    = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_    = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_    = TensorDesc(problem.c_dtype_, c_layout);
                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::AddGelu;

                gemm_op->epilogue_op_ = TensorOperation::AddGelu;

                gemm_op->ds_layout_ = {c_layout};
                gemm_op->ds_dtype_  = {problem.c_dtype_};

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
    else if (problem.epilogue_op_ == TensorOperation::AddMultiply) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = problem.operation_kind_;
                if (problem.operation_kind_ == GemmOperationKind::Gemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemmMultipleD_Xdl_CShuffle;
                }
                else if (problem.operation_kind_ == GemmOperationKind::BatchGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceBatchedGemmMultiD_Xdl;
                }
                else if (problem.operation_kind_ == GemmOperationKind::SplitKGemm) {
                    gemm_op->kernel_type_ = GemmKernelType::DeviceGemm_Xdl_CShuffleV3R1;
                }
                else {
                    LI_THROW(Unimplemented("unsuuport gemm operation kind"));
                }

                gemm_op->a_tensor_desc_    = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_    = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_    = TensorDesc(problem.c_dtype_, c_layout);
                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::AddMultiply;

                gemm_op->epilogue_op_ = TensorOperation::AddMultiply;

                gemm_op->ds_layout_ = {c_layout, c_layout};
                gemm_op->ds_dtype_  = {problem.c_dtype_, problem.c_dtype_};

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
}

// GemmPermuteM2N3 + any epilogue
void CreateGemmPermuteOperations(const GemmProblem& problem)
{
    auto a_layout = std::get<0>(problem.GetLayout());
    auto b_layout = std::get<1>(problem.GetLayout());
    auto c_layout = std::get<2>(problem.GetLayout());

    // GemmPermuteM2N3
    if (problem.epilogue_op_ == TensorOperation::PassThrough) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = GemmOperationKind::GemmPermuteM2N3;
                gemm_op->kernel_type_    = GemmKernelType::DeviceBatchedContractionMultipleD_Xdl_CShuffle;

                gemm_op->a_tensor_desc_    = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_    = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_    = TensorDesc(problem.c_dtype_, c_layout);
                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::PassThrough;

                gemm_op->epilogue_op_ = TensorOperation::PassThrough;

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
    // GemmPermuteM2N3 + bias
    else if (problem.epilogue_op_ == TensorOperation::Add) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = GemmOperationKind::GemmPermuteM2N3;
                gemm_op->kernel_type_    = GemmKernelType::DeviceBatchedContractionMultipleD_Xdl_CShuffle;

                gemm_op->a_tensor_desc_    = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_    = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_    = TensorDesc(problem.c_dtype_, c_layout);
                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_ = TensorOperation::PassThrough;
                gemm_op->b_element_op_ = TensorOperation::PassThrough;
                gemm_op->c_element_op_ = TensorOperation::Add;

                gemm_op->epilogue_op_ = TensorOperation::Add;

                gemm_op->gemm_specialization_ = GetGemmSpec(problem.m_,
                                                            problem.n_,
                                                            problem.k_,
                                                            gemm_op->tile_desc_.m_per_block_,
                                                            gemm_op->tile_desc_.n_per_block_,
                                                            gemm_op->tile_desc_.k_per_block_);
                return gemm_op;
            },
            a_layout,
            b_layout);
    }
}

template<typename F>
void CreateXlopsAttnKernel(F                      f,
                           const LayoutType&      a_layout,
                           const LayoutType&      b_layout,
                           const TensorOperation& masking_spec)
{
    std::vector<AttnTileDesc> tile_descriptions = {
        // clang-format off
//  Block|  MPer|  NPer|  KPer| Gemm1NPer| Gemm1NPer|  AK1|  BK1|  B1K1|  MPer| NPer| MXdl| NXdl| Gemm1NXdl|
//   Size| Block| Block| Block|     Block|     Block|     |     |      |   XDL|  XDL|  Per|  Per|       Per|
//       |      |      |      |          |          |     |     |      |      |     | Wave| Wave|      Wave|
//       |      |      |      |          |          |     |     |      |      |     |     |     |          |
  {   256,   128,   128,    32,        64,        32,    8,    8,     2,    32,   32,    1,    4,         2}  // clang-format on
    };

    std::vector<BlockTransferDesc> a_block_descriptions = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1}  // clang-format on
    };

    std::vector<BlockTransferDesc> b_block_descriptions = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1}  // clang-format on
    };

    std::vector<BlockTransferDesc> b1_block_descriptions = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    {16, 16, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              2,         0}  // clang-format on
    };

    std::vector<MaskedCBlockTransferDesc> c_block_descriptions = {
        // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
  {1, 2,              {1, 32, 1, 8},               8, masking_spec}
        // clang-format on
    };

    // check size
    LI_ENFORCE_EQ(
        tile_descriptions.size(),
        a_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), a_block_descriptions.size()));
    LI_ENFORCE_EQ(
        tile_descriptions.size(),
        b_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), b_block_descriptions.size()));
    LI_ENFORCE_EQ(
        tile_descriptions.size(),
        c_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), c_block_descriptions.size()));

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < tile_descriptions.size(); i++) {
        std::shared_ptr<GemmOperation> gemm_operation = std::make_shared<GemmOperation>();
        gemm_operation->attn_tile_desc_               = tile_descriptions[i];
        gemm_operation->a_block_transfer_             = a_block_descriptions[i];
        gemm_operation->b_block_transfer_             = b_block_descriptions[i];
        gemm_operation->mask_c_block_transfer_        = c_block_descriptions[i];
        gemm_operation->b1_block_transfer_            = b1_block_descriptions[i];

        auto all = f(gemm_operation);
        emitters_ptr->Append(std::move(all));
    }
}

// fmha
template<typename F>
void CreateFmhaFwdKernel(F f)
{
    std::vector<FmhaTileDesc> tile_desc_vec = {
        // clang-format off
    {128, 64, 16, 32, 32, 32, 2, 1, 1,  2, 1, 1,  32, 32, 16, 32, 32, 16},
    {128, 128, 32, 256, 32,  256,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16},
    {128, 64,  32, 64,  32,  64,   4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16},
    {128, 128, 32, 128, 32,  128,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16},
    {128, 128, 32, 256, 32,  256,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16}  // clang-format on
    };

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < tile_desc_vec.size(); i++) {
        std::shared_ptr<FmhaFwdOperation> fmha_op = std::make_shared<FmhaFwdOperation>();
        fmha_op->tile_desc_                       = tile_desc_vec[i];
        auto all                                  = f(fmha_op);
        emitters_ptr->Append(std::move(all));
    }
}

template<typename F>
void CreateFmhaFwdAppendKVKernel(F f)
{
    std::vector<FmhaAppendKVTileDesc> tile_desc_vec = {
        // clang-format off
    {64, 64,  32,  32}  // clang-format on
    };

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < tile_desc_vec.size(); i++) {
        std::shared_ptr<FmhaFwdAppendKVOperation> fmha_op = std::make_shared<FmhaFwdAppendKVOperation>();
        fmha_op->tile_desc_                               = tile_desc_vec[i];
        auto all                                          = f(fmha_op);
        emitters_ptr->Append(std::move(all));
    }
}

template<typename F>
void CreateFmhaFwdSplitKVKernel(F f)
{
    std::vector<FmhaTileDesc> splitkv_tile_desc_vec = {
        // clang-format off
    {128, 64, 16, 32, 32, 32, 2, 1, 1,  2, 1, 1,  32, 32, 16, 32, 32, 16}  // clang-format on
    };

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < splitkv_tile_desc_vec.size(); i++) {
        std::shared_ptr<FmhaFwdSplitKVOperation> fmha_op = std::make_shared<FmhaFwdSplitKVOperation>();
        fmha_op->tile_desc_                              = splitkv_tile_desc_vec[i];
        auto all                                         = f(fmha_op);
        emitters_ptr->Append(std::move(all));
    }
}

template<typename F>
void CreateFmhaFwdSplitKVCombineKernel(F f)
{
    std::vector<FmhaSplitKVCompbineTileDesc> splitkv_combine_tile_desc_vec = {
        // clang-format off
    {128, 32}  // clang-format on
    };

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < splitkv_combine_tile_desc_vec.size(); i++) {
        std::shared_ptr<FmhaFwdSplitKVCombineOperation> fmha_op = std::make_shared<FmhaFwdSplitKVCombineOperation>();
        fmha_op->tile_desc_                                     = splitkv_combine_tile_desc_vec[i];
        auto all                                                = f(fmha_op);
        emitters_ptr->Append(std::move(all));
    }
}

void CreateFmhaOperations(const FmhaProblem& problem)
{
    if (problem.operation_kind_ == FmhaOperationKind::Fwd) {
        return CreateFmhaFwdKernel([&](const std::shared_ptr<FmhaFwdOperation>& fmha_op) {
            fmha_op->operation_kind_ = FmhaOperationKind::Fwd;
            fmha_op->epilogue_op_    = TensorOperation::PassThrough;

            fmha_op->operation_mode_ = problem.operation_mode_;

            fmha_op->dtype_       = problem.dtype_;
            fmha_op->mask_type_   = problem.mask_type_;
            fmha_op->window_size_ = problem.window_size_;
            fmha_op->bias_enum_   = problem.bias_enum_;

            fmha_op->block_per_cu_ = -1;

            fmha_op->pipeline_ = BlockFmhaPipelineEnum::QRKSVS_ASYNC;

            fmha_op->is_pad_q_seq_len_ = fmha_op->pipeline_ == BlockFmhaPipelineEnum::QRKSVS_ASYNC ?
                                             true :
                                             problem.q_seq_len_ % fmha_op->tile_desc_.bm0_ != 0;
            fmha_op->is_pad_kv_seq_len_ =
                fmha_op->pipeline_ == BlockFmhaPipelineEnum::QRKSVS_ASYNC ?
                    true :
                    (problem.kv_seq_len_ == 0) || (problem.kv_seq_len_ % fmha_op->tile_desc_.bn0_ != 0);
            fmha_op->is_pad_v_head_dim_  = fmha_op->pipeline_ == BlockFmhaPipelineEnum::QRKSVS_ASYNC ?
                                               true :
                                               problem.v_head_dim_ % fmha_op->tile_desc_.bn1_ != 0;
            fmha_op->is_pad_qk_head_dim_ = fmha_op->pipeline_ == BlockFmhaPipelineEnum::QRKSVS_ASYNC ?
                                               true :
                                               problem.qk_head_dim_ % fmha_op->tile_desc_.bk0_max_ != 0;

            // usually headdim_q and headdim_v are same, consider them together to
            // determine whether to do padding saving some compiling time
            fmha_op->is_pad_qkv_head_dim_ = (fmha_op->is_pad_v_head_dim_ || fmha_op->is_pad_qk_head_dim_);

            return fmha_op;
        });
    }
    else if (problem.operation_kind_ == FmhaOperationKind::FwdAppendKV) {
        return CreateFmhaFwdAppendKVKernel([&](const std::shared_ptr<FmhaFwdAppendKVOperation>& fmha_op) {
            fmha_op->operation_kind_ = FmhaOperationKind::FwdAppendKV;
            fmha_op->epilogue_op_    = TensorOperation::PassThrough;

            fmha_op->operation_mode_ = problem.operation_mode_;
            fmha_op->dtype_          = problem.dtype_;
            fmha_op->rope_type_      = problem.rope_type_;
            fmha_op->is_paged_kv_    = problem.paged_block_size_ > 0 ? true : false;

            fmha_op->is_pad_q_seq_len_ = problem.operation_mode_ == FmhaOperationMode::Batch ?
                                             problem.q_seq_len_ % fmha_op->tile_desc_.bs_ != 0 :
                                             true;
            fmha_op->is_pad_kv_seq_len_ =
                problem.operation_mode_ == FmhaOperationMode::Batch ?
                    (problem.kv_seq_len_ == 0) || (problem.kv_seq_len_ % fmha_op->tile_desc_.bsk_ != 0) :
                    true;
            fmha_op->is_pad_v_head_dim_  = problem.operation_mode_ == FmhaOperationMode::Batch ?
                                               problem.v_head_dim_ % fmha_op->tile_desc_.bdv_ != 0 :
                                               true;
            fmha_op->is_pad_qk_head_dim_ = problem.operation_mode_ == FmhaOperationMode::Batch ?
                                               problem.qk_head_dim_ % fmha_op->tile_desc_.bd_ != 0 :
                                               true;

            // usually headdim_q and headdim_v are same, consider them together to
            // determine whether to do padding saving some compiling time
            fmha_op->is_pad_qkv_head_dim_ = (fmha_op->is_pad_v_head_dim_ || fmha_op->is_pad_qk_head_dim_);

            return fmha_op;
        });
    }
    else if (problem.operation_kind_ == FmhaOperationKind::FwdSplitKV) {
        return CreateFmhaFwdSplitKVKernel([&](const std::shared_ptr<FmhaFwdSplitKVOperation>& fmha_op) {
            fmha_op->operation_kind_ = FmhaOperationKind::FwdSplitKV;
            fmha_op->epilogue_op_    = TensorOperation::PassThrough;

            fmha_op->operation_mode_ = problem.operation_mode_;
            fmha_op->dtype_          = problem.dtype_;
            fmha_op->mask_type_      = problem.mask_type_;
            fmha_op->window_size_    = problem.window_size_;
            fmha_op->bias_enum_      = problem.bias_enum_;
            fmha_op->is_store_lse_   = problem.num_splits_ > 1 ? true : false;

            fmha_op->has_uneven_splits_ =
                problem.operation_mode_ == FmhaOperationMode::Batch ?
                    problem.kv_seq_len_ % (problem.num_splits_ * fmha_op->tile_desc_.bn0_) != 0 :
                    true;

            fmha_op->is_merge_num_head_groups_seq_len_q_ =
                problem.q_max_seq_len_ == 1 && problem.kv_num_heads_ < problem.q_num_heads_ ? true : false;

            fmha_op->is_pad_q_seq_len_ = problem.operation_mode_ == FmhaOperationMode::Batch ?
                                             problem.q_seq_len_ % fmha_op->tile_desc_.bm0_ != 0 :
                                             true;
            fmha_op->is_pad_kv_seq_len_ =
                problem.operation_mode_ == FmhaOperationMode::Batch ? fmha_op->has_uneven_splits_ : true;

            fmha_op->is_pad_v_head_dim_  = problem.v_head_dim_ % fmha_op->tile_desc_.bn1_ != 0;
            fmha_op->is_pad_qk_head_dim_ = problem.qk_head_dim_ % fmha_op->tile_desc_.bk0_max_ != 0;

            // usually headdim_q and headdim_v are same, consider them together to
            // determine whether to do padding saving some compiling time
            fmha_op->is_pad_qkv_head_dim_ = (fmha_op->is_pad_v_head_dim_ || fmha_op->is_pad_qk_head_dim_);

            return fmha_op;
        });
    }
    else if (problem.operation_kind_ == FmhaOperationKind::FwdSplitKVCombine) {
        return CreateFmhaFwdSplitKVCombineKernel([&](const std::shared_ptr<FmhaFwdSplitKVCombineOperation>& fmha_op) {
            fmha_op->operation_kind_ = FmhaOperationKind::FwdSplitKVCombine;
            fmha_op->epilogue_op_    = TensorOperation::PassThrough;

            fmha_op->operation_mode_ = problem.operation_mode_;
            fmha_op->dtype_          = problem.dtype_;
            fmha_op->hdim_           = problem.v_head_dim_;

            auto get_log_max_splits = [&]() {
                if (problem.num_splits_ <= 8) {
                    return 3;
                }
                else if (problem.num_splits_ <= 16) {
                    return 4;
                }
                else if (problem.num_splits_ <= 32) {
                    return 5;
                }
                else if (problem.num_splits_ <= 64) {
                    return 6;
                }
                else if (problem.num_splits_ <= 128) {
                    return 7;
                }
                else {
                    throw std::runtime_error("num_splits_ is too large");
                }
            };

            fmha_op->log_max_splits_ = get_log_max_splits();

            fmha_op->is_pad_q_seq_len_  = problem.operation_mode_ == FmhaOperationMode::Batch ?
                                              problem.q_seq_len_ % fmha_op->tile_desc_.bm0_ != 0 :
                                              true;
            fmha_op->is_pad_v_head_dim_ = problem.v_head_dim_ % fmha_op->tile_desc_.bn1_ != 0;

            return fmha_op;
        });
    }
    else {
        LI_THROW(Unimplemented("unsupport fmha operation kind"));
    }
}

// Layer norm + any epilogue
template<typename F>
void CreateXlopsLayerNormKernel(F f)
{
    std::vector<NormTileDesc> tile_desc_vec = {
        // clang-format off
    {1,  1,  8,  8,  8},
    {1,  1,  4,  16, 4},
    {1,  1,  4,  64, 1},
    {1,  1,  4,  16, 8},
    {1,  1,  4,  64, 2}  // clang-format on
    };

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < tile_desc_vec.size(); i++) {
        std::shared_ptr<NormOperation> layer_norm_operation = std::make_shared<NormOperation>();
        layer_norm_operation->tile_desc_                    = tile_desc_vec[i];
        auto all                                            = f(layer_norm_operation);
        emitters_ptr->Append(std::move(all));
    }
}

void CreateNormOperations(const NormProblem& problem)
{
    if (problem.epilogue_op_ == TensorOperation::PassThrough) {
        return CreateXlopsLayerNormKernel([&](const std::shared_ptr<NormOperation>& norm_op) {
            norm_op->operation_kind_ = problem.operation_kind_;
            norm_op->epilogue_op_    = TensorOperation::PassThrough;

            norm_op->is_add_bias_ = problem.is_add_bias_;
            norm_op->fused_add_   = problem.fused_add_;
            norm_op->fused_quant_ = problem.fused_quant_;

            norm_op->x_dtype_            = problem.x_dtype_;
            norm_op->y_dtype_            = problem.y_dtype_;
            norm_op->smooth_scale_dtype_ = problem.smooth_scale_dtype_;
            norm_op->y_scale_dtype_      = problem.y_scale_dtype_;

            return norm_op;
        });
    }
}

// embedding
template<typename F>
void CreateEmbeddingKernel(F f, const int64_t embedding_dims)
{

    int64_t row_v_size = std::gcd(8, embedding_dims / 256);

    std::vector<EmbeddingTileDesc> embedding_tile_descriptions = {
        // clang-format off
    // BlockSize | DimClusterSize | RowClusterSize | DimPerBlock | RowPerBlock | DimThreadSize | RowVectorSize |
        {    256,               1,              256,            1, embedding_dims,          1,                  row_v_size}
        // clang-format on
    };

    auto emitters_ptr = Emitters::GetInstance();

    for (int i = 0; i < embedding_tile_descriptions.size(); i++) {
        // check
        if (embedding_dims
                % (embedding_tile_descriptions[i].row_cluster_size_ * embedding_tile_descriptions[i].row_vector_size_)
            != 0) {
            LI_THROW(InvalidArgument("embedding_dims {} is not divisible by row_cluster_size_ {} * row_vector_size_ {}",
                                     embedding_dims,
                                     embedding_tile_descriptions[i].row_cluster_size_,
                                     embedding_tile_descriptions[i].row_vector_size_));
        }

        std::shared_ptr<EmbeddingOperation> embedding_operation = std::make_shared<EmbeddingOperation>();
        embedding_operation->tile_desc_                         = embedding_tile_descriptions[i];
        auto all                                                = f(embedding_operation);
        emitters_ptr->Append(std::move(all));
    }
}

void CreateEmbeddingOperations(const EmbeddingProblem& problem)
{
    if (problem.epilogue_op_ == TensorOperation::PassThrough) {
        return CreateEmbeddingKernel(
            [&](const std::shared_ptr<EmbeddingOperation>& embedding_op) {
                embedding_op->operation_kind_        = EmbeddingOperationKind::SparseEmbedding;
                embedding_op->epilogue_op_           = TensorOperation::PassThrough;
                embedding_op->embedding_kernel_type_ = EmbeddingKernelType::DeviceSparseEmbedding;

                embedding_op->vocab_size_     = problem.vocab_size_;
                embedding_op->embedding_dims_ = problem.embedding_dims_;
                embedding_op->num_elements_   = 1;

                embedding_op->emb_dtype_   = problem.emb_dtype_;
                embedding_op->index_dtype_ = problem.index_dtype_;
                embedding_op->gamma_dtype_ = problem.gamma_dtype_;
                embedding_op->beta_dtype_  = problem.beta_dtype_;
                embedding_op->acc_dtype_   = problem.acc_dtype_;
                embedding_op->y_dtype_     = problem.y_dtype_;

                return embedding_op;
            },
            problem.embedding_dims_);
    }
    else if (problem.epilogue_op_ == TensorOperation::AddAddLayerNorm) {
        return CreateEmbeddingKernel(
            [&](const std::shared_ptr<EmbeddingOperation>& embedding_op) {
                embedding_op->operation_kind_        = EmbeddingOperationKind::SparseEmbedding;
                embedding_op->epilogue_op_           = TensorOperation::AddAddLayerNorm;
                embedding_op->embedding_kernel_type_ = EmbeddingKernelType::DeviceSparseEmbeddingLayerNorm;

                embedding_op->vocab_size_              = problem.vocab_size_;
                embedding_op->type_vocab_size_         = problem.type_vocab_size_;
                embedding_op->max_position_embeddings_ = problem.max_position_embeddings_;
                embedding_op->embedding_dims_          = problem.embedding_dims_;
                embedding_op->num_elements_            = 3;

                embedding_op->emb_dtype_   = problem.emb_dtype_;
                embedding_op->index_dtype_ = problem.index_dtype_;
                embedding_op->gamma_dtype_ = problem.gamma_dtype_;
                embedding_op->beta_dtype_  = problem.beta_dtype_;
                embedding_op->acc_dtype_   = problem.acc_dtype_;
                embedding_op->y_dtype_     = problem.y_dtype_;

                return embedding_op;
            },
            problem.embedding_dims_);
    }
}

void GenerateXldopsCKKernel(const GenOperationKind&                                                      gen_op_kind,
                            const std::variant<GemmProblem, NormProblem, EmbeddingProblem, FmhaProblem>& problem)
{

    if (gen_op_kind == GenOperationKind::Gemm) {
        auto gemm_op_kind = std::get<GemmProblem>(problem).operation_kind_;
        if (gemm_op_kind == GemmOperationKind::Gemm || gemm_op_kind == GemmOperationKind::BatchGemm
            || gemm_op_kind == GemmOperationKind::SplitKGemm) {
            CreateGemmOperations(std::get<GemmProblem>(problem));
        }
        else if (gemm_op_kind == GemmOperationKind::GemmPermuteM2N3) {
            CreateGemmPermuteOperations(std::get<GemmProblem>(problem));
        }
    }
    else if (gen_op_kind == GenOperationKind::Norm) {
        CreateNormOperations(std::get<NormProblem>(problem));
    }
    else if (gen_op_kind == GenOperationKind::Embedding) {
        CreateEmbeddingOperations(std::get<EmbeddingProblem>(problem));
    }
    else if (gen_op_kind == GenOperationKind::Fmha) {
        CreateFmhaOperations(std::get<FmhaProblem>(problem));
    }
    else {
        LI_THROW(Unimplemented("lightinfer unsupport operation kind"));
    }
}

void GenerateCKKernelDispatch(const GenOperationKind&                                                      gen_op_kind,
                              const std::variant<GemmProblem, NormProblem, EmbeddingProblem, FmhaProblem>& problem)
{
    if (Target::Instance()->GetTargetDeviceArch() == "xdl") {
        GenerateXldopsCKKernel(gen_op_kind, problem);
    }
    else {
        LI_THROW(Unavailable("lightinfer unsupport device arch {} now", Target::Instance()->GetTargetDeviceArch()));
    }
}

}  // namespace lightinfer