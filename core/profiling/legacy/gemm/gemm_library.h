#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace flashck {

namespace legacy{

/**
 * @enum GemmKind
 * @brief Defines the types of GEMM operations supported
 */
enum class GemmKind : int {
    Gemm          = 0,  ///< Standard general matrix multiplication
    GemmMultipleD = 1,  ///< GEMM with multiple D tensors
    GemmMX       = 2,  ///< GEMM with microscaling datatype
    COUNT  // Used for iteration and validation
};

/**
 * @struct GemmInfo
 * @brief Information about GEMM operation types
 */
struct GemmInfo {
    std::string name;        ///< Human-readable name of the GEMM operation
    std::string device_tag;  ///< Template tag for device implementation
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from GEMM types to their information
 */
static const std::unordered_map<GemmKind, GemmInfo> g_gemm_info_map = {
    {GemmKind::Gemm, {"gemm", "ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3", "G"}},
    {GemmKind::GemmMultipleD,
     {"gemm_multiple_d", "ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle", "GM"}},
    {GemmKind::GemmMX, {"gemm_mx", "ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3", "GMX"}}};

/**
 * @enum LayoutType
 * @brief Defines tensor layout types for GEMM operations
 */
enum class LayoutType : int {
    ColumnMajor = 0,  ///< Column-major layout (Fortran-style)
    RowMajor    = 1,  ///< Row-major layout (C-style)
    COUNT             // Used for validation
};

/**
 * @struct LayoutInfo
 * @brief Information about tensor layouts
 */
struct LayoutInfo {
    std::string name;        ///< Full descriptive name
    std::string class_tag;   ///< C++ class template tag
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from layout types to their information
 */
static const std::unordered_map<LayoutType, LayoutInfo> g_layout_map = {
    {LayoutType::ColumnMajor, {"column_major", "ck::tensor_layout::gemm::ColumnMajor", "N"}},
    {LayoutType::RowMajor, {"row_major", "ck::tensor_layout::gemm::RowMajor", "T"}},
};

/**
 * @enum BlockGemmPipelineVersion
 * @brief Defines different GEMM pipeline versions
 */
enum class BlockGemmPipelineVersion : int {
    V1 = 0,  ///< Naive implementation
    V2 = 1,  ///< Memory optimized
    V3 = 2,  ///< Compute optimized
    V4 = 3,  ///< Compute optimized with double LDS buffer
    V5 = 4,  ///< Compute optimized with double global prefetch register buffer
    COUNT    // Used for validation
};

/**
 * @struct PipelineVersionInfo
 * @brief Information about pipeline versions
 */
struct PipelineVersionInfo {
    std::string name;         ///< Full descriptive name
    std::string version_tag;  ///< Version identifier
    std::string short_name;   ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from pipeline versions to their information
 */
static const std::unordered_map<BlockGemmPipelineVersion, PipelineVersionInfo> g_pipeline_version_map = {
    {BlockGemmPipelineVersion::V1, {"naive", "v1", "N"}},
    {BlockGemmPipelineVersion::V2, {"memory_optimized", "v2", "M"}},
    {BlockGemmPipelineVersion::V3, {"compute_optimized", "v3", "C"}},
    {BlockGemmPipelineVersion::V4, {"compute_double_lds", "v4", "CDL"}},
    {BlockGemmPipelineVersion::V5, {"compute_double_global", "v5", "CDG"}},
};

/**
 * @enum BlockGemmPipelineScheduler
 * @brief Defines different GEMM pipeline schedulers
 */
enum class BlockGemmPipelineScheduler : int {
    Intrawave = 0,  ///< Intra-wave scheduling
    Interwave = 1,  ///< Inter-wave scheduling
    COUNT           // Used for validation
};

/**
 * @struct SchedulerInfo
 * @brief Information about pipeline schedulers
 */
struct SchedulerInfo {
    std::string name;        ///< Full descriptive name
    std::string class_tag;   ///< C++ class template tag
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from schedulers to their information
 */
static const std::unordered_map<BlockGemmPipelineScheduler, SchedulerInfo> g_scheduler_map = {
    {BlockGemmPipelineScheduler::Intrawave, {"intrawave", "ck::BlockGemmPipelineScheduler::Intrawave", "IW"}},
    {BlockGemmPipelineScheduler::Interwave, {"interwave", "ck::BlockGemmPipelineScheduler::Interwave", "EW"}},
};

/**
 * @enum EpilogueType
 * @brief Defines different epilogue operations for GEMM
 */
enum class EpilogueType : int {
    PassThrough = 0,  ///< No additional operation
    Add         = 1,  ///< Element-wise addition
    AddGelu     = 2,  ///< Addition followed by GELU activation
    AddTanh     = 3,  ///< Addition followed by Tanh activation
    AddSiLU     = 4,  ///< Addition followed by SiLU activation
    AddMultiply = 5,  ///< Addition followed by multiplication
    AddAdd      = 6,  ///< Double addition operation
    COUNT             // Used for validation
};

/**
 * @struct EpilogueInfo
 * @brief Information about epilogue operations
 */
struct EpilogueInfo {
    std::string name;        ///< Full descriptive name
    std::string class_tag;   ///< C++ class template tag
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from epilogue types to their information
 */
static const std::unordered_map<EpilogueType, EpilogueInfo> g_epilogue_map = {
    {EpilogueType::PassThrough, {"pass_through", "ck::tensor_operation::element_wise::PassThrough", "PT"}},
    {EpilogueType::Add, {"add", "ck::tensor_operation::element_wise::Add", "A"}},
    {EpilogueType::AddGelu, {"add_gelu", "ck::tensor_operation::element_wise::AddFastGelu", "AFG"}},
    {EpilogueType::AddTanh, {"add_tanh", "ck::tensor_operation::element_wise::AddTanh", "AT"}},
    {EpilogueType::AddSiLU, {"add_silu", "ck::tensor_operation::element_wise::AddSiLU", "AS"}},
    {EpilogueType::AddMultiply, {"add_multiply", "ck::tensor_operation::element_wise::AddMultiply", "AM"}},
    {EpilogueType::AddAdd, {"add_add", "ck::tensor_operation::element_wise::AddAdd", "AA"}}};

enum class GemmSpecialization : int {
    Default    = 0,  ///< No specialization (standard GEMM)
    MPadding   = 1,  ///< Specialization for M dimension padding
    NPadding   = 2,  ///< Specialization for N dimension padding
    KPadding   = 3,  ///< Specialization for K dimension padding
    MNPadding  = 4,  ///< Specialization for M and N dimensions padding
    MKPadding  = 5,  ///< Specialization for M and K dimensions padding
    NKPadding  = 6,  ///< Specialization for N and K dimensions padding
    MNKPadding = 7,  ///< Specialization for M, N, and K dimensions padding
    COUNT            // Used for validation
};

/**
 * @struct GemmSpecializationInfo
 * @brief Information about GEMM specialization types
 */
struct GemmSpecializationInfo {
    std::string name;        ///< Full descriptive name
    std::string class_tag;   ///< C++ class template tag
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from GEMM specialization types to their information
 */
static const std::unordered_map<GemmSpecialization, GemmSpecializationInfo> g_gemm_specialization_map = {
    {GemmSpecialization::Default, {"default", "ck::tensor_operation::device::GemmSpecialization::Default", "D"}},
    {GemmSpecialization::MPadding, {"m_padding", "ck::tensor_operation::device::GemmSpecialization::MPadding", "M"}},
    {GemmSpecialization::NPadding, {"n_padding", "ck::tensor_operation::device::GemmSpecialization::NPadding", "N"}},
    {GemmSpecialization::KPadding, {"k_padding", "ck::tensor_operation::device::GemmSpecialization::KPadding", "K"}},
    {GemmSpecialization::MNPadding,
     {"mn_padding", "ck::tensor_operation::device::GemmSpecialization::MNPadding", "MN"}},
    {GemmSpecialization::MKPadding,
     {"mk_padding", "ck::tensor_operation::device::GemmSpecialization::MKPadding", "MK"}},
    {GemmSpecialization::NKPadding,
     {"nk_padding", "ck::tensor_operation::device::GemmSpecialization::NKPadding", "NK"}},
    {GemmSpecialization::MNKPadding,
     {"mnk_padding", "ck::tensor_operation::device::GemmSpecialization::MNKPadding", "MNK"}}};

// Legacy compatibility maps (deprecated, use g_gemm_specialization_map instead)
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

// ====================== Utility Functions ======================

/**
 * @brief Gets the name string for a GEMM operation kind
 * @param kind The GEMM operation kind to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetGemmKindName(GemmKind kind)
{
    auto it = g_gemm_info_map.find(kind);
    return it != g_gemm_info_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the device tag for a GEMM operation kind
 * @param kind The GEMM operation kind to query
 * @return The device tag string, or "unknown" if not found
 */
inline std::string GetGemmKindDeviceTag(GemmKind kind)
{
    auto it = g_gemm_info_map.find(kind);
    return it != g_gemm_info_map.end() ? it->second.device_tag : "unknown";
}

/**
 * @brief Gets the short name for a GEMM operation kind
 * @param kind The GEMM operation kind to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetGemmKindShortName(GemmKind kind)
{
    auto it = g_gemm_info_map.find(kind);
    return it != g_gemm_info_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for a layout type
 * @param layout The layout type to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetLayoutName(LayoutType layout)
{
    auto it = g_layout_map.find(layout);
    return it != g_layout_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the class tag for a layout type
 * @param layout The layout type to query
 * @return The class tag string, or "unknown" if not found
 */
inline std::string GetLayoutClassTag(LayoutType layout)
{
    auto it = g_layout_map.find(layout);
    return it != g_layout_map.end() ? it->second.class_tag : "unknown";
}

/**
 * @brief Gets the short name for a layout type
 * @param layout The layout type to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetLayoutShortName(LayoutType layout)
{
    auto it = g_layout_map.find(layout);
    return it != g_layout_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for a pipeline version
 * @param version The pipeline version to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetPipelineVersionName(BlockGemmPipelineVersion version)
{
    auto it = g_pipeline_version_map.find(version);
    return it != g_pipeline_version_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the version tag for a pipeline version
 * @param version The pipeline version to query
 * @return The version tag string, or "unknown" if not found
 */
inline std::string GetPipelineVersionTag(BlockGemmPipelineVersion version)
{
    auto it = g_pipeline_version_map.find(version);
    return it != g_pipeline_version_map.end() ? it->second.version_tag : "unknown";
}

/**
 * @brief Gets the BlockGemmPipelineVersion enum from a string (name, version_tag, or short_name)
 * @param str The string to match
 * @return The corresponding BlockGemmPipelineVersion, or BlockGemmPipelineVersion::COUNT if not found
 */
inline BlockGemmPipelineVersion GetPipelineVersionFromString(const std::string& str)
{
    for (const auto& [ver, info] : g_pipeline_version_map) {
        if (info.name == str || info.version_tag == str || info.short_name == str) {
            return ver;
        }
    }
    return BlockGemmPipelineVersion::COUNT;
}

/**
 * @brief Gets the short name for a pipeline version
 * @param version The pipeline version to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetPipelineVersionShortName(BlockGemmPipelineVersion version)
{
    auto it = g_pipeline_version_map.find(version);
    return it != g_pipeline_version_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for a scheduler type
 * @param scheduler The scheduler type to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetSchedulerName(BlockGemmPipelineScheduler scheduler)
{
    auto it = g_scheduler_map.find(scheduler);
    return it != g_scheduler_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the short name for a scheduler type
 * @param scheduler The scheduler type to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetSchedulerShortName(BlockGemmPipelineScheduler scheduler)
{
    auto it = g_scheduler_map.find(scheduler);
    return it != g_scheduler_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the class tag for a scheduler type
 * @param scheduler The scheduler type to query
 * @return The class tag string, or "unknown" if not found
 */
inline std::string GetSchedulerClassTag(BlockGemmPipelineScheduler scheduler)
{
    auto it = g_scheduler_map.find(scheduler);
    return it != g_scheduler_map.end() ? it->second.class_tag : "unknown";
}

/**
 * @brief Gets the BlockGemmPipelineScheduler enum from a string (name, class_tag, or short_name)
 * @param str The string to match
 * @return The corresponding BlockGemmPipelineScheduler, or BlockGemmPipelineScheduler::COUNT if not found
 */
inline BlockGemmPipelineScheduler GetPipelineSchedulerFromString(const std::string& str)
{
    for (const auto& [sched, info] : g_scheduler_map) {
        if (info.name == str || info.class_tag == str || info.short_name == str) {
            return sched;
        }
    }
    return BlockGemmPipelineScheduler::COUNT;
}


/**
 * @brief Gets the name string for an epilogue type
 * @param epilogue The epilogue type to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetEpilogueName(EpilogueType epilogue)
{
    auto it = g_epilogue_map.find(epilogue);
    return it != g_epilogue_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the class tag for an epilogue type
 * @param epilogue The epilogue type to query
 * @return The class tag string, or "unknown" if not found
 */
inline std::string GetEpilogueClassTag(EpilogueType epilogue)
{
    auto it = g_epilogue_map.find(epilogue);
    return it != g_epilogue_map.end() ? it->second.class_tag : "unknown";
}

/**
 * @brief Gets the short name for an epilogue type
 * @param epilogue The epilogue type to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetEpilogueShortName(EpilogueType epilogue)
{
    auto it = g_epilogue_map.find(epilogue);
    return it != g_epilogue_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for a GEMM specialization type
 * @param spec The GEMM specialization type to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetGemmSpecializationName(GemmSpecialization spec)
{
    auto it = g_gemm_specialization_map.find(spec);
    return it != g_gemm_specialization_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the class tag for a GEMM specialization type
 * @param spec The GEMM specialization type to query
 * @return The class tag string, or "unknown" if not found
 */
inline std::string GetGemmSpecializationClassTag(GemmSpecialization spec)
{
    auto it = g_gemm_specialization_map.find(spec);
    return it != g_gemm_specialization_map.end() ? it->second.class_tag : "unknown";
}

/**
 * @brief Gets the short name for a GEMM specialization type
 * @param spec The GEMM specialization type to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetGemmSpecializationShortName(GemmSpecialization spec)
{
    auto it = g_gemm_specialization_map.find(spec);
    return it != g_gemm_specialization_map.end() ? it->second.short_name : "unknown";
}

// ====================== Validation Functions ======================

/**
 * @brief Validates if a GEMM operation kind is valid
 * @param kind The GEMM operation kind to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidGemmKind(GemmKind kind)
{
    return static_cast<int>(kind) >= 0 && static_cast<int>(kind) < static_cast<int>(GemmKind::COUNT);
}

/**
 * @brief Validates if a layout type is valid
 * @param layout The layout type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidLayoutType(LayoutType layout)
{
    return static_cast<int>(layout) >= 0 && static_cast<int>(layout) < static_cast<int>(LayoutType::COUNT);
}

/**
 * @brief Validates if a pipeline version is valid
 * @param version The pipeline version to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidPipelineVersion(BlockGemmPipelineVersion version)
{
    return static_cast<int>(version) >= 0
           && static_cast<int>(version) < static_cast<int>(BlockGemmPipelineVersion::COUNT);
}

/**
 * @brief Validates if a scheduler type is valid
 * @param scheduler The scheduler type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidScheduler(BlockGemmPipelineScheduler scheduler)
{
    return static_cast<int>(scheduler) >= 0
           && static_cast<int>(scheduler) < static_cast<int>(BlockGemmPipelineScheduler::COUNT);
}

/**
 * @brief Validates if an epilogue type is valid
 * @param epilogue The epilogue type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidEpilogueType(EpilogueType epilogue)
{
    return static_cast<int>(epilogue) >= 0 && static_cast<int>(epilogue) < static_cast<int>(EpilogueType::COUNT);
}

/**
 * @brief Validates if a GEMM specialization type is valid
 * @param spec The GEMM specialization type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidGemmSpecialization(GemmSpecialization spec)
{
    return static_cast<int>(spec) >= 0 && static_cast<int>(spec) < static_cast<int>(GemmSpecialization::COUNT);
}

} // namespace legacy
}  // namespace flashck