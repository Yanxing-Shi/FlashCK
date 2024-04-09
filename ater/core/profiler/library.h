#pragma once
#include <unordered_map>

#include "ater/core/utils/dtype.h"

namespace ater {

// // Gen op kind
enum class GenOperationKind {
    Gemm      = 0,
    LayerNorm = 1,
};

// static const std::unordered_map<GenOperationKind, std::string> g_gen_operation_kind_names{
//     {GenOperationKind::Gemm, "gemm"}, {GenOperationKind::LayerNorm, "layernorm"}};

/*--------------------------------gemm-------------------------------------------------------------*/
enum class GemmSpecialization {
    // Gemm
    Default    = 0,
    MPadding   = 1,
    NPadding   = 2,
    KPadding   = 3,
    MNPadding  = 4,
    MKPadding  = 5,
    NKPadding  = 6,
    MNKPadding = 7
};

static const std::unordered_map<GemmSpecialization, std::string> g_gemm_specialization_tag = {
    {GemmSpecialization::Default, "ck::tensor_operation::device::GemmSpecialization::Default"},
    {GemmSpecialization::MPadding, "ck::tensor_operation::device::GemmSpecialization::MPadding"},
    {GemmSpecialization::NPadding, "ck::tensor_operation::device::GemmSpecialization::NPadding"},
    {GemmSpecialization::KPadding, "ck::tensor_operation::device::GemmSpecialization::KPadding"},
    {GemmSpecialization::MNPadding, "ck::tensor_operation::device::GemmSpecialization::MNPadding"},
    {GemmSpecialization::MKPadding, "ck::tensor_operation::device::GemmSpecialization::MKPadding"},
    {GemmSpecialization::NKPadding, "ck::tensor_operation::device::GemmSpecialization::NKPadding"},
    {GemmSpecialization::MNKPadding, "ck::tensor_operation::device::GemmSpecialization::MNKPadding"}};

static const std::unordered_map<GemmSpecialization, std::string> g_short_gemm_spec_names{
    {GemmSpecialization::Default, "D"},
    {GemmSpecialization::MPadding, "M"},
    {GemmSpecialization::NPadding, "N"},
    {GemmSpecialization::KPadding, "K"},
    {GemmSpecialization::MNPadding, "MN"},
    {GemmSpecialization::MKPadding, "MK"},
    {GemmSpecialization::NKPadding, "NK"},
    {GemmSpecialization::MNKPadding, "MNK"}};

static const std::unordered_map<std::string, GemmSpecialization> g_gemm_spec_names{
    {"", GemmSpecialization::Default},
    {"M", GemmSpecialization::MPadding},
    {"N", GemmSpecialization::NPadding},
    {"K", GemmSpecialization::KPadding},
    {"MN", GemmSpecialization::MNPadding},
    {"MK", GemmSpecialization::MKPadding},
    {"NK", GemmSpecialization::NKPadding},
    {"MNK", GemmSpecialization::MNKPadding}};

enum class KernelType {
    DeviceGemmXdl_CShuffle                           = 0,
    DeviceGemmMultipleD_Xdl_CShuffle                 = 1,
    DeviceBatchedGemmXdl                             = 2,
    DeviceBatchedGemmCPermuteXdl                     = 3,
    DeviceGemmBiasCPermute_Xdl                       = 4,
    DeviceBatchedContractionMultipleD_Xdl_CShuffle   = 5,
    DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle        = 6,
    DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle = 7,
    DeviceBatchedGemmMultiD_Xdl                      = 8,
};

static const std::unordered_map<KernelType, std::string> g_kernel_tag = {
    {KernelType::DeviceGemmXdl_CShuffle, "ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle"},
    {KernelType::DeviceGemmMultipleD_Xdl_CShuffle, "ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle"},
    {KernelType::DeviceBatchedGemmXdl, "ck::tensor_operation::device::DeviceBatchedGemmXdl"},
    {KernelType::DeviceBatchedGemmCPermuteXdl, "ck::tensor_operation::device::DeviceBatchedGemmEPermuteXdl"},
    {KernelType::DeviceGemmBiasCPermute_Xdl, "ck::tensor_operation::device::DeviceGemmBiasEPermute_Xdl"},
    {KernelType::DeviceBatchedContractionMultipleD_Xdl_CShuffle,
     "ck::tensor_operation::device::DeviceBatchedContractionMultipleD_Xdl_CShuffle"},
    {KernelType::DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle,
     "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle"},
    {KernelType::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle,
     "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle"},
    {KernelType::DeviceBatchedGemmMultiD_Xdl, "ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl"}};

/*--------------------------------------Layer norm--------------------------------------------------------------*/

/*-------------------------------------common utils----------------------------------------------------------*/
static const std::unordered_map<DataType, std::string> g_short_data_type_names{{DataType::FLOAT32, "f32"},
                                                                               {DataType::FLOAT16, "f16"}};

static const std::unordered_map<DataType, std::string> g_data_type_tag{{DataType::FLOAT32, "float"},
                                                                       {DataType::FLOAT16, "ck::half_t"}};

enum class LayoutType {
    ColumnMajor = 0,
    RowMajor    = 1,
};

static std::unordered_map<LayoutType, std::string> g_layout_tag{
    {LayoutType::ColumnMajor, "ck::tensor_layout::gemm::ColumnMajor"},
    {LayoutType::RowMajor, "ck::tensor_layout::gemm::RowMajor"}};

static std::unordered_map<LayoutType, std::string> g_short_layout_names{{LayoutType::ColumnMajor, "N"},
                                                                        {LayoutType::RowMajor, "T"}};

enum class OperationKind {
    Gemm                        = 0,
    GemmPermute                 = 1,
    BatchGemm                   = 2,
    BatchGemmPermute            = 3,
    SplitKGemm                  = 4,
    Grouped                     = 5,
    BatchGemmSoftmaxGemm        = 6,
    BatchGemmSoftmaxGemmPermute = 7,
    GemmPermuteM2N3             = 8,
    GemmPermuteM3N2             = 9
};

static std::unordered_map<OperationKind, std::string> g_operation_kind_names{
    {OperationKind::Gemm, "gemm"},
    {OperationKind::GemmPermute, "gemm_permute"},
    {OperationKind::BatchGemm, "batch_gemm"},
    {OperationKind::BatchGemmPermute, "batch_gemm_permute"}};

enum class TensorOperation {
    PassThrough = 0,
    Add         = 1,
    AddFastGelu = 2,
    AddTanh     = 3,
    CausalMask  = 4,
    UnDefined   = 5
};

static std::unordered_map<TensorOperation, std::string> g_tensor_operation_tag{
    {TensorOperation::PassThrough, "ck::tensor_operation::element_wise::PassThrough"},
    {TensorOperation::Add, "ck::tensor_operation::element_wise::Add"},
    {TensorOperation::AddFastGelu, "ck::tensor_operation::element_wise::AddFastGelu"},
    {TensorOperation::AddTanh, "ck::tensor_operation::element_wise::AddTanh"},
    {TensorOperation::CausalMask, "True"}};

static std::unordered_map<TensorOperation, std::string> g_short_tensor_operation_names{
    {TensorOperation::PassThrough, "PT"},
    {TensorOperation::Add, "A"},
    {TensorOperation::AddFastGelu, "AFG"},
    {TensorOperation::AddTanh, "AT"},
    {TensorOperation::CausalMask, "CM"}};

class TensorDesc {
public:
    TensorDesc() = default;
    TensorDesc(DataType element_type, LayoutType layout_type): element_(element_type), layout_(layout_type) {}

    DataType   element_;
    LayoutType layout_;
};
}  // namespace ater