#include "ater/core/profiler/generator.h"

#include "ater/core/profiler/library.h"
#include "ater/core/utils/enforce.h"

namespace ater {

template<typename F>
void CreateXlopsGemmKernel(F f, const LayoutType& a_layout, const LayoutType& b_layout)
{

    std::vector<TileDesc> tile_descriptions = {
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
    std::vector<BlockTransferDesc> a_block_descriptions_rowmajor = {
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

    std::vector<BlockTransferDesc> a_block_descriptions_colmajor = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
        // clang-format on
        {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 8, 1},
        {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
        {{4, 32, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 8, 1},
        {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
        {{4, 32, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 8, 1},
        {{4, 32, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
        {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 2, 8, 1},
        {{4, 64, 1}, {0, 2, 1}, {0, 2, 1}, 1, 1, 8, 1},
    };

    std::vector<BlockTransferDesc> b_block_descriptions_rowmajor = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
  {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              4,              8,         1},
  {    {4, 32, 1},     {0, 2, 1},     {0, 2, 1},              1,              4,              8,         1},
  {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
  {    {4, 32, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
  {    {4, 32, 1},     {0, 2, 1},     {0, 2, 1},              1,              4,              8,         1},
  {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              1,              8,         1},
  {    {4, 64, 1},     {0, 2, 1},     {0, 2, 1},              1,              2,              8,         1},
        // clang-format on
    };

    std::vector<BlockTransferDesc> b_block_descriptions_colmajor = {
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

    const auto a_block_descriptions =
        (a_layout == LayoutType::RowMajor) ? a_block_descriptions_rowmajor : a_block_descriptions_colmajor;
    const auto b_block_descriptions =
        (b_layout == LayoutType::RowMajor) ? b_block_descriptions_rowmajor : b_block_descriptions_colmajor;

    // check size
    ATER_ENFORCE_EQ(
        tile_descriptions.size(),
        a_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), a_block_descriptions.size()));
    ATER_ENFORCE_EQ(
        tile_descriptions.size(),
        b_block_descriptions.size(),
        Fatal("tile dim size {} not equal to block dim size{}", tile_descriptions.size(), b_block_descriptions.size()));
    ATER_ENFORCE_EQ(
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

void CreateGemmOperations(const GemmProblem& problem)
{

    auto a_layout = std::get<0>(problem.GetLayout());
    auto b_layout = std::get<1>(problem.GetLayout());
    auto c_layout = std::get<2>(problem.GetLayout());

    // gemm
    if (problem.epilogue_op_ == TensorOperation::PassThrough) {
        return CreateXlopsGemmKernel(
            [&](const std::shared_ptr<GemmOperation>& gemm_op) {
                gemm_op->operation_kind_ = OperationKind::Gemm;
                gemm_op->extra_kind_     = TensorOperation::PassThrough;
                gemm_op->kernel_type_    = KernelType::DeviceGemmMultipleD_Xdl_CShuffle;

                gemm_op->a_tensor_desc_    = TensorDesc(problem.a_dtype_, a_layout);
                gemm_op->b_tensor_desc_    = TensorDesc(problem.b_dtype_, b_layout);
                gemm_op->c_tensor_desc_    = TensorDesc(problem.c_dtype_, c_layout);
                gemm_op->accumulator_type_ = problem.acc_dtype_;

                gemm_op->a_element_op_     = TensorOperation::PassThrough;
                gemm_op->b_element_op_     = TensorOperation::PassThrough;
                gemm_op->epilogue_functor_ = TensorOperation::PassThrough;

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

void GenerateXldopsCKKernel(const GenOperationKind& gen_op_kind, const GemmProblem& gemm_problem)
{
    if (gen_op_kind == GenOperationKind::Gemm) {
        CreateGemmOperations(gemm_problem);
    }
    else {
        ATER_THROW(Unimplemented("{}", "ater unsupport operation kind"));
    }
}

// void GenerateWmmaCKernel(std::shared_ptr<Emitters>& emitter_ptr) {}

void GenerateCKKernelDispatch(const GenOperationKind& gen_op_kind, const GemmProblem& gemm_problem)
{
    if (Target::Instance()->GetTargetDeviceArch() == "xdl") {
        GenerateXldopsCKKernel(gen_op_kind, gemm_problem);
    }
    else {
        ATER_THROW(Unavailable("ater unsupport device arch {} now", Target::Instance()->GetTargetDeviceArch()));
    }
}

}  // namespace ater