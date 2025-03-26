#pragma once

#include <unordered_map>

#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

// Gen op kind
enum class GenOperationKind {
    Gemm      = 0,
    Embedding = 1,
    Norm      = 2,
    Fmha      = 3,
};

static const std::unordered_map<GenOperationKind, std::string> g_gen_operation_kind_names{
    {GenOperationKind::Gemm, "gemm"},
    {GenOperationKind::Embedding, "embedding"},
    {GenOperationKind::Norm, "norm"},
    {GenOperationKind::Fmha, "fmha"}};

/*--------------------------------embedding------------------------------------------------------------*/
enum class EmbeddingOperationKind {
    SparseEmbedding          = 0,
    SparseEmbeddingLayerNorm = 1,
};

static const std::unordered_map<EmbeddingOperationKind, std::string> g_embedding_operation_kind_names{
    {EmbeddingOperationKind::SparseEmbedding, "sparse_embedding"},
    {EmbeddingOperationKind::SparseEmbeddingLayerNorm, "sparse_embedding_layernorm"}};

enum class EmbeddingKernelType {
    DeviceSparseEmbedding          = 0,
    DeviceSparseEmbeddingLayerNorm = 1
};

static const std::unordered_map<EmbeddingKernelType, std::string> g_embedding_kernel_tag = {
    {EmbeddingKernelType::DeviceSparseEmbedding,
     "ck::tensor_operation::device::DeviceSparseEmbeddingsForwardLayernorm"},
    {EmbeddingKernelType::DeviceSparseEmbeddingLayerNorm,
     "ck::tensor_operation::device::DeviceSparseEmbeddingsForwardLayernorm"}};

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

enum class GemmOperationKind {
    Gemm                        = 0,
    GemmPermute                 = 1,
    BatchGemm                   = 2,
    BatchGemmPermute            = 3,
    SplitKGemm                  = 4,
    Grouped                     = 5,
    BatchGemmSoftmaxGemm        = 6,
    BatchGemmSoftmaxGemmPermute = 7,
    GemmPermuteM2N3             = 8,
    GemmPermuteM3N2             = 9,
};

static std::unordered_map<GemmOperationKind, std::string> g_gemm_operation_kind_names{
    {GemmOperationKind::Gemm, "gemm"},
    {GemmOperationKind::GemmPermute, "gemm_permute"},
    {GemmOperationKind::BatchGemm, "batch_gemm"},
    {GemmOperationKind::BatchGemmPermute, "batch_gemm_permute"},
    {GemmOperationKind::SplitKGemm, "split_k_gemm"},
    {GemmOperationKind::Grouped, "grouped"},
    {GemmOperationKind::BatchGemmSoftmaxGemm, "batch_gemm_softmax_gemm"},
    {GemmOperationKind::BatchGemmSoftmaxGemmPermute, "batch_gemm_softmax_gemm_permute"},
    {GemmOperationKind::GemmPermuteM2N3, "gemm_permute_m2n3"},
    {GemmOperationKind::GemmPermuteM3N2, "gemm_permute_m3n2"}};

enum class GemmKernelType {
    DeviceGemmXdl_CShuffle                           = 0,
    DeviceGemmMultipleD_Xdl_CShuffle                 = 1,
    DeviceBatchedGemmXdl                             = 2,
    DeviceBatchedGemmCPermuteXdl                     = 3,
    DeviceGemmBiasCPermute_Xdl                       = 4,
    DeviceBatchedContractionMultipleD_Xdl_CShuffle   = 5,
    DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle        = 6,
    DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle = 7,
    DeviceBatchedGemmMultiD_Xdl                      = 8,
    DeviceGemmXdlSplitKCShuffle                      = 9,
    DeviceGemm_Xdl_CShuffleV3R1                      = 10
};

static const std::unordered_map<GemmKernelType, std::string> g_kernel_tag = {
    {GemmKernelType::DeviceGemmXdl_CShuffle, "ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle"},
    {GemmKernelType::DeviceGemmMultipleD_Xdl_CShuffle,
     "ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle"},
    {GemmKernelType::DeviceBatchedGemmXdl, "ck::tensor_operation::device::DeviceBatchedGemmXdl"},
    {GemmKernelType::DeviceBatchedGemmCPermuteXdl, "ck::tensor_operation::device::DeviceBatchedGemmEPermuteXdl"},
    {GemmKernelType::DeviceGemmBiasCPermute_Xdl, "ck::tensor_operation::device::DeviceGemmBiasEPermute_Xdl"},
    {GemmKernelType::DeviceBatchedContractionMultipleD_Xdl_CShuffle,
     "ck::tensor_operation::device::DeviceBatchedContractionMultipleD_Xdl_CShuffle"},
    {GemmKernelType::DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle,
     "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle"},
    {GemmKernelType::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle,
     "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle"},
    {GemmKernelType::DeviceBatchedGemmMultiD_Xdl, "ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl"},
    {GemmKernelType::DeviceGemmXdlSplitKCShuffle, "ck::tensor_operation::device::DeviceGemmXdlSplitKCShuffle"},
    {GemmKernelType::DeviceGemm_Xdl_CShuffleV3R1, "ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3R1"}};

/*---------------------------------------norm-----------------------------------------------------------*/
enum class NormOperationKind {
    LayerNorm = 0,
    RMSNorm   = 1,
};

static const std::unordered_map<NormOperationKind, std::string> g_norm_operation_kind_names_map{
    {NormOperationKind::LayerNorm, "layer_norm"}, {NormOperationKind::RMSNorm, "rms_norm"}};

static const std::unordered_map<NormOperationKind, std::string> g_norm_operation_problem_tag_map{
    {NormOperationKind::LayerNorm, "Layernorm2dFwdPipelineProblem"},
    {NormOperationKind::RMSNorm, "Rmsnorm2dFwdPipelineProblem"}};

static const std::unordered_map<NormOperationKind, std::string> g_norm_operation_trait_tag_map{
    {NormOperationKind::LayerNorm, "Layernorm2dFwdTraits"}, {NormOperationKind::RMSNorm, "Rmsnorm2dFwdTraits"}};

static const std::unordered_map<NormOperationKind, std::string> g_norm_operation_fwd_tag_map{
    {NormOperationKind::LayerNorm, "Layernorm2dFwd"}, {NormOperationKind::RMSNorm, "Rmsnorm2dFwd"}};

static const std::unordered_map<NormOperationKind, std::string> g_norm_operation_pass_tag_map{
    {NormOperationKind::LayerNorm, "Layernorm2dFwdPipelineTwoPass"},
    {NormOperationKind::RMSNorm, "Layernorm2dFwdPipelineOnePass"}};

enum class NormBiasEnum {
    NO_BIAS = 0,
    // add bias before fused add
    ADD_BIAS = 1,
};

static const std::unordered_map<NormBiasEnum, std::string> g_tile_layer_norm_operation_kind_names_map{
    {NormBiasEnum::NO_BIAS, "no_bias"}, {NormBiasEnum::ADD_BIAS, "add_bias"}};

static const std::unordered_map<NormBiasEnum, std::string> g_tile_layer_norm_operation_kind_short_names_map{
    {NormBiasEnum::NO_BIAS, "nb"}, {NormBiasEnum::ADD_BIAS, "ab"}};

enum class FusedAddEnum {
    NO_ADD        = 0,
    PRE_ADD_STORE = 1,  // fused add before layernorm and store result to global
    PRE_ADD       = 2,  //  fused add before layernorm, but not store result
};

static const std::unordered_map<FusedAddEnum, std::string> g_fused_add_enum_str_map = {
    {FusedAddEnum::NO_ADD, "no_add"},
    {FusedAddEnum::PRE_ADD_STORE, "pre_add_store"},
    {FusedAddEnum::PRE_ADD, "pre_add"}};

enum class FusedQuantEnum {
    NO_SWEEP             = 0,
    SMOOTH_DYNAMIC_QUANT = 1,  // smooth oulier + rowwise quant, need input x-scale and store y_scale
    DYNAMIC_QUANT        = 2,  // rowwise quant, store out a y-scale
};

static const std::unordered_map<FusedQuantEnum, std::string> g_fused_quant_enum_str_map = {
    {FusedQuantEnum::NO_SWEEP, "no_sweep"},
    {FusedQuantEnum::SMOOTH_DYNAMIC_QUANT, "smooth_dynamic_quant"},
    {FusedQuantEnum::DYNAMIC_QUANT, "dynamic_quant"}};

/*----------------------------------fmha----------------------------------------*/
enum class FmhaOperationMode {
    Batch = 0,
    Group = 1,
};

static const std::unordered_map<FmhaOperationMode, std::string> g_fmha_operation_mode_name_map{
    {FmhaOperationMode::Batch, "batch"}, {FmhaOperationMode::Group, "group"}};

enum class FmhaOperationKind {
    Fwd               = 0,
    FwdAppendKV       = 1,
    FwdSplitKV        = 2,
    FwdSplitKVCombine = 3,
};

static const std::unordered_map<FmhaOperationKind, std::string> g_fmha_kind_names_map{
    {FmhaOperationKind::Fwd, "fmha_fwd"},
    {FmhaOperationKind::FwdAppendKV, "fmha_fwd_appendkv"},
    {FmhaOperationKind::FwdSplitKV, "fmha_fwd_splitkv"},
    {FmhaOperationKind::FwdSplitKVCombine, "fmha_fwd_splitkv_combine"}};

// quant
enum class QuantMode {
    None  = 0,
    Auto  = 1,
    Quant = 2,
};

static const std::unordered_map<QuantMode, std::string> g_quant_mode_map{
    {QuantMode::None, "no"}, {QuantMode::Auto, "Auto"}, {QuantMode::Quant, "Quant"}};

// mask
enum class GenericAttentionMaskEnum {
    NO_MASK = 0,

    // below enum could be causal, or sliding window
    MASK_FROM_TOP_LEFT     = 1,
    MASK_FROM_BOTTOM_RIGHT = 2,

    // this enum maybe not used by xformer/FA, since it's hard to
    // specify left/right window for varlen case. put it here for
    // debug purpose
    MASK_GENERIC,
};

static const std::unordered_map<GenericAttentionMaskEnum, std::string> g_generic_attention_mask_names_map{
    {GenericAttentionMaskEnum::NO_MASK, "no"},
    {GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT, "from_top_left"},
    {GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT, "from_top_right"},
    {GenericAttentionMaskEnum::MASK_GENERIC, "generic"}};

static const std::unordered_map<GenericAttentionMaskEnum, std::string> g_generic_attention_mask_short_names_map{
    {GenericAttentionMaskEnum::NO_MASK, "no"},
    {GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT, "tl"},
    {GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT, "br"},
    {GenericAttentionMaskEnum::MASK_GENERIC, "ge"}};

struct MaskEnumInfo {
    GenericAttentionMaskEnum type_;
    int64_t                  sliding_window_size_;
};

// enum class MaskImplType {
//     SimplifiedNo   = 0,
//     SimplifiedMask = 1,
// };

// static const std::unordered_map<MaskImplType, std::string> g_simplified_mask_type_tag{
//     {MaskImplType::SimplifiedNo, "ck_tile::SimplifiedGenericAttentionMask<false>"},
//     {MaskImplType::SimplifiedMask, "ck_tile::SimplifiedGenericAttentionMask<true>"}};

enum class BiasEnum {
    NO_BIAS          = 0,
    ELEMENTWISE_BIAS = 1,  // attention bias, each elements add to the result of Q*K(after scale)
    ALIBI            = 2,  // bias computed with position encoding, applied after scale
};

static const std::unordered_map<BiasEnum, std::string> g_bias_enum_tag = {
    {BiasEnum::NO_BIAS, "ck_tile::BlockAttentionBiasEnum::NO_BIAS"},
    {BiasEnum::ELEMENTWISE_BIAS, "ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS"},
    {BiasEnum::ALIBI, "ck_tile::BlockAttentionBiasEnum::ALIBI"}};

static const std::unordered_map<BiasEnum, std::string> g_bias_enum_names_map = {
    {BiasEnum::NO_BIAS, "no"}, {BiasEnum::ELEMENTWISE_BIAS, "elementwise"}, {BiasEnum::ALIBI, "alibi"}};

static const std::unordered_map<BiasEnum, std::string> g_bias_enum_short_names_map = {
    {BiasEnum::NO_BIAS, "nb"}, {BiasEnum::ELEMENTWISE_BIAS, "eb"}, {BiasEnum::ALIBI, "ab"}};

struct BiasEnumInfo {
    BiasEnum type_;
    /*
     * simple dispatch logic
     *
     * if type == elementwise_bias:
     *      if rank_info == 0:
     *           bias is 1*1*s*s
     *      elif rank_info == 1:
     *           bias is 1*h*s*s
     *      elif rank_info == 2:
     *           bias is b*h*s*s
     *
     * elif type == alibi:
     *       if rank_info == 0:
     *           alibi in 1*h
     *       elif rank_info == 1:
     *           alibi in b*h
     */
    int rank_info_;

    std::ostream& operator<<(std::ostream& os)
    {
        if (type_ == BiasEnum::NO_BIAS)
            os << "n";
        else if (type_ == BiasEnum::ELEMENTWISE_BIAS) {
            os << "e";
            if (rank_info_ != 0) {
                os << "[" << rank_info_ << "]";
            }
        }
        else if (type_ == BiasEnum::ALIBI) {
            os << "alibi";
            if (rank_info_ != 0) {
                os << "[" << rank_info_ << "]";
            }
        }

        return os;
    }
};

enum class RopeEnum {
    NONE         = 0,
    INTERLEAVED  = 1,  // combine dimensions 0 & 1, 2 & 3, etc
    HALF_ROTATED = 2,  // combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1, etc
};

static const std::unordered_map<RopeEnum, std::string> g_rope_enum_tag = {
    {RopeEnum::NONE, "ck_tile::RotaryEmbeddingEnum::NONE"},
    {RopeEnum::INTERLEAVED, "ck_tile::RotaryEmbeddingEnum::INTERLEAVED"},
    {RopeEnum::HALF_ROTATED, "ck_tile::RotaryEmbeddingEnum::HALF_ROTATED"}};

static const std::unordered_map<RopeEnum, std::string> g_rope_enum_short_names_map = {
    {RopeEnum::NONE, "none"}, {RopeEnum::INTERLEAVED, "inter"}, {RopeEnum::HALF_ROTATED, "half"}};

// pipeline
enum class BlockFmhaPipelineEnum {
    QRKSVS            = 0,
    QRKSVS_ASYNC      = 1,
    QR_NWARP_SSHUFFLE = 2,
    QSKSVS            = 3,
};

static const std::unordered_map<BlockFmhaPipelineEnum, std::string> g_block_fmha_fwd_pipeline_map = {
    {BlockFmhaPipelineEnum::QRKSVS, "ck_tile::BlockFmhaPipelineQRKSVS"},
    {BlockFmhaPipelineEnum::QRKSVS_ASYNC, "ck_tile::BlockFmhaPipelineQRKSVSAsync"}};

static const std::unordered_map<BlockFmhaPipelineEnum, std::string> g_block_fmha_fwd_splitkv_pipeline_map = {
    {BlockFmhaPipelineEnum::QRKSVS, "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS"},
    {BlockFmhaPipelineEnum::QR_NWARP_SSHUFFLE, "ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS"},
    {BlockFmhaPipelineEnum::QRKSVS_ASYNC, "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVSAsync"}};

static const std::unordered_map<BlockFmhaPipelineEnum, std::string> g_block_fmha_pipeline_short_name_map = {
    {BlockFmhaPipelineEnum::QRKSVS, "qr"}, {BlockFmhaPipelineEnum::QRKSVS_ASYNC, "qr_async"}};

enum class InitMethod {
    UniformRandomInt          = 0,
    NormalizedRandomInt       = 1,
    UniformRandomFloat        = 2,
    NormalizedRandomFloat     = 3,
    TrigFloat                 = 4,
    UniformFloat8Quantization = 5,
};

static const std::unordered_map<InitMethod, std::string> g_init_method_short_names_map = {
    {InitMethod::UniformRandomInt, "uri"},
    {InitMethod::NormalizedRandomInt, "nri"},
    {InitMethod::UniformRandomFloat, "urf"},
    {InitMethod::NormalizedRandomFloat, "nrf"},
    {InitMethod::TrigFloat, "tf"},
    {InitMethod::UniformFloat8Quantization, "uf8q"},
};

/*-------------------------------------common utils----------------------------------------------------------*/
enum class LayoutType {
    ColumnMajor = 0,
    RowMajor    = 1,
};

static std::unordered_map<LayoutType, std::string> g_layout_tag{
    {LayoutType::ColumnMajor, "ck::tensor_layout::gemm::ColumnMajor"},
    {LayoutType::RowMajor, "ck::tensor_layout::gemm::RowMajor"}};

static std::unordered_map<LayoutType, std::string> g_short_layout_names{{LayoutType::ColumnMajor, "N"},
                                                                        {LayoutType::RowMajor, "T"}};

enum class TensorOperation {
    PassThrough                  = 0,
    Add                          = 1,
    AddGelu                      = 2,
    AddTanh                      = 3,
    AddSiLU                      = 4,
    MaskDisabled                 = 5,
    MaskUpperTriangleFromTopLeft = 6,
    MaskLowerTriangleFromTopLeft = 7,
    AddMultiply                  = 8,
    AddAdd                       = 9,
    AddAddLayerNorm              = 10,  // fuse device op

    // tile layer norm
    PreAddStore  = 11,
    PreAdd       = 12,
    DynamicQuant = 13,
};

static const std::unordered_map<TensorOperation, std::string> g_tensor_operation_names{
    {TensorOperation::PassThrough, "PassThrough"},
    {TensorOperation::Add, "Add"},
    {TensorOperation::AddGelu, "AddGelu"},
    {TensorOperation::AddTanh, "AddTanh"},
    {TensorOperation::AddSiLU, "AddSiLU"},
    {TensorOperation::MaskDisabled, "MaskDisabled"},
    {TensorOperation::MaskUpperTriangleFromTopLeft, "MaskUpperTriangleFromTopLeft"},
    {TensorOperation::MaskLowerTriangleFromTopLeft, "MaskLowerTriangleFromTopLeft"},
    {TensorOperation::AddMultiply, "AddMultiply"},
    {TensorOperation::AddAdd, "AddAdd"},
    {TensorOperation::AddAddLayerNorm, "AddAddLayerNorm"}};

static std::unordered_map<TensorOperation, std::string> g_tensor_operation_tag{
    {TensorOperation::PassThrough, "ck::tensor_operation::element_wise::PassThrough"},
    {TensorOperation::Add, "ck::tensor_operation::element_wise::Add"},
    {TensorOperation::AddGelu, "ck::tensor_operation::element_wise::AddFastGelu"},
    {TensorOperation::AddTanh, "ck::tensor_operation::element_wise::AddTanh"},
    {TensorOperation::AddSiLU, "ck::tensor_operation::element_wise::AddSiLU"},
    {TensorOperation::MaskDisabled, "ck::tensor_operation::device::MaskingSpecialization::MaskDisabled"},
    {TensorOperation::MaskUpperTriangleFromTopLeft,
     "ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft"},
    {TensorOperation::MaskLowerTriangleFromTopLeft,
     "ck::tensor_operation::device::MaskingSpecialization::MaskLowerTriangleFromTopLeft"},
    {TensorOperation::AddMultiply, "ck::tensor_operation::element_wise::AddMultiply"},
    {TensorOperation::AddAdd, "ck::tensor_operation::element_wise::AddAdd"}};

static std::unordered_map<TensorOperation, std::string> g_short_tensor_operation_names_map{
    {TensorOperation::PassThrough, "PT"},
    {TensorOperation::Add, "A"},
    {TensorOperation::AddGelu, "AFG"},
    {TensorOperation::AddTanh, "AT"},
    {TensorOperation::AddSiLU, "AS"},
    {TensorOperation::MaskDisabled, "D"},
    {TensorOperation::MaskUpperTriangleFromTopLeft, "U"},
    {TensorOperation::MaskLowerTriangleFromTopLeft, "L"},
    {TensorOperation::AddMultiply, "AM"},
    {TensorOperation::AddAdd, "AA"},
    {TensorOperation::AddAddLayerNorm, "AAL"}};

// /*----------------------------------------------------mask--------------------------------------------------------*/
// enum class MaskingSpec {
//     MaskDisabled                 = 0,
//     MaskUpperTriangleFromTopLeft = 1,
//     MaskLowerTriangleFromTopLeft = 2,
// };

// static std::unordered_map<MaskingSpec, std::string> g_masking_specialization_tag{
//     {MaskingSpec::MaskDisabled, "ck::tensor_operation::device::MaskingSpecialization::MaskDisabled"},
//     {MaskingSpec::MaskUpperTriangleFromTopLeft,
//      "ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft"},
//     {MaskingSpec::MaskLowerTriangleFromTopLeft,
//      "ck::tensor_operation::device::MaskingSpecialization::MaskLowerTriangleFromTopLeft"}};

// static std::unordered_map<MaskingSpec, std::string> g_short_masking_spec_names{
//     {MaskingSpec::MaskDisabled, "D"},
//     {MaskingSpec::MaskUpperTriangleFromTopLeft, "U"},
//     {MaskingSpec::MaskLowerTriangleFromTopLeft, "L"}};

class TensorDesc {
public:
    TensorDesc() = default;
    TensorDesc(DataType element_type, LayoutType layout_type): element_(element_type), layout_(layout_type) {}

    DataType   element_;
    LayoutType layout_;
};

}  // namespace lightinfer