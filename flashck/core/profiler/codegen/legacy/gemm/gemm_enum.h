#pragma once

namespace flashck {

enum class GemmSpecialization {
    Default    = 0,
    MPadding   = 1,
    NPadding   = 2,
    KPadding   = 3,
    MNPadding  = 4,
    MKPadding  = 5,
    NKPadding  = 6,
    MNKPadding = 7
};

template<GemmSpecialization Spec>
struct GemmSpecializationTraits;

template<GemmSpecialization Spec>
constexpr const char* GetGemmSpecializationTag()
{
    return GemmSpecializationTraits<Spec>::tag;
}

template<GemmSpecialization Spec>
constexpr const char* GetShortSpecializationName()
{
    return GemmSpecializationTraits<Spec>::short_name;
}

#define DEFINE_GEMM_SPEC_TRAITS(EnumValue, Tag, ShortName)                                                             \
    template<>                                                                                                         \
    struct GemmSpecializationTraits<GemmSpecialization::EnumValue> {                                                   \
        static constexpr const char* tag        = "ck::tensor_operation::device::GemmSpecialization::" + Tag;          \
        static constexpr const char* short_name = ShortName;                                                           \
    };

DEFINE_GEMM_SPEC_TRAITS(Default, "Default", "NoPadding")
DEFINE_GEMM_SPEC_TRAITS(MPadding, "MPadding", "M_Pad")
DEFINE_GEMM_SPEC_TRAITS(NPadding, "NPadding", "N_Pad")
DEFINE_GEMM_SPEC_TRAITS(KPadding, "KPadding", "K_Pad")
DEFINE_GEMM_SPEC_TRAITS(MNPadding, "MNPadding", "MN_Pad")
DEFINE_GEMM_SPEC_TRAITS(MKPadding, "MKPadding", "MK_Pad")
DEFINE_GEMM_SPEC_TRAITS(NKPadding, "NKPadding", "NK_Pad")
DEFINE_GEMM_SPEC_TRAITS(MNKPadding, "MNKPadding", "MNK_Pad")

enum class LayoutType {
    ColumnMajor = 0,
    RowMajor    = 1,
};

enum class GemmOperationKind {
    Gemm            = 0,
    BatchGemm       = 1,
    SplitKGemm      = 2,
    Grouped         = 3,
    GemmPermuteM2N3 = 4,
    GemmPermuteM3N2 = 5
};

enum class GemmKernelType {
    DeviceGemmXdl_CShuffle                         = 0,
    DeviceGemmMultipleD_Xdl_CShuffle               = 1,
    DeviceBatchedGemmXdl                           = 2,
    DeviceBatchedContractionMultipleD_Xdl_CShuffle = 3,
    DeviceGemmXdlSplitKCShuffle                    = 4
};

enum class TensorOperation {
    PassThrough = 0,
    Add         = 1,
    AddGelu     = 2,
    AddTanh     = 3,
    AddSiLU     = 4,
    AddMultiply = 8,
    AddAdd      = 9
};

}  // namespace flashck