#pragma once

#include <string>
#include <unordered_map>

namespace flashck {

/**
 * @enum GemmKind
 * @brief Supported GEMM operation types
 */
enum class GemmKind {
    Gemm = 0,        ///< Standard GEMM
    GemmMultiD = 1,  ///< Multi-dimensional GEMM
    Flatmm = 2,      ///< Flat matrix multiplication
    BatchGemm = 3,   ///< Batched GEMM
    GroupGemm = 4    ///< Grouped GEMM
};

struct GemmKindInfo
{
    std::string name;
    std::string short_name;
    std::string tag;
};

static const std::unordered_map<GemmKind, GemmKindInfo> g_gemm_kind_info_map{
    {GemmKind::Gemm, {"gemm", "Gemm", "ck_tile::GemmKernel"}},
    {GemmKind::GemmMultiD, {"gemm_multi_d", "GemmMD", "ck_tile::GemmKernelMultiD"}},
    {GemmKind::Flatmm, {"flatmm", "Flatmm", "ck_tile::FlatmmKernel"}},
    {GemmKind::BatchGemm, {"batch_gemm", "BatchGemm", "ck_tile::BatchGemmKernel"}},
    {GemmKind::GroupGemm, {"group_gemm", "GroupGemm", "ck_tile::GroupedGemmKernel"}}
};


/**
 * @enum ElementwiseKind
 * @brief Supported elementwise operations for GEMM
 */
enum class ElementwiseKind {
    PassThrough = 0, ///< No-op
    Add = 1,         ///< Addition
    MultiplyMultiply = 2 ///< Multiplication
};

struct ElementwiseKindInfo
{
    std::string name;
    std::string short_name;
    std::string tag;
};

static const std::unordered_map<ElementwiseKind, ElementwiseKindInfo> g_elementwise_kind_info_map{
    {ElementwiseKind::PassThrough, {"PassThrough", "PT", "ck_tile::PassThrough"}},
    {ElementwiseKind::Add, {"Add", "A", "ck_tile::Add"}},
    {ElementwiseKind::MultiplyMultiply, {"MultiplyMultiply", "MM", "ck_tile::MultiplyMultiply"}}
};


/**
 * @enum PipelineEnum
 * @brief Pipeline types for GEMM kernel
 */
enum class PipelineEnum {
    Mem = 0,
    Compute_V3 = 1,
    Compute_V4 = 2,
    Compute_V5 = 3,
    Preshuffle = 4
};


struct PipelineEnumInfo
{
    std::string name;
    std::string short_name;
    std::string base_tag;
    std::string main_tag;
};

// Pipeline type to tag mapping
static const std::unordered_map<PipelineEnum, PipelineEnumInfo> g_pipeline_info_map{
    {PipelineEnum::Mem, {"GemmPipelineAgBgCrMem", "Mem", "ck_tile::BaseGemmPipelineAgBgCrMem", "ck_tile::GemmPipelineAgBgCrMem"}},
    {PipelineEnum::Compute_V3, {"GemmPipelineAgBgCrCompV3", "CompV3", "ck_tile::BaseGemmPipelineAgBgCrCompV3", "ck_tile::GemmPipelineAgBgCrCompV3"}},
    {PipelineEnum::Compute_V4, {"GemmPipelineAgBgCrCompV4", "CompV4", "ck_tile::BaseGemmPipelineAgBgCrCompV4", "ck_tile::GemmPipelineAgBgCrCompV4"}},
    {PipelineEnum::Compute_V5, {"GemmPipelineAgBgCrCompV5", "CompV5", "ck_tile::BaseGemmPipelineAgBgCrCompV5", "ck_tile::GemmPipelineAgBgCrCompV5"}},
    {PipelineEnum::Preshuffle, {"WeightPreshufflePipelineAGmemBGmemCRegV1", "Preshuffle", "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV1", "ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV1"}}
};

/**
 * @enum SchedulerEnum
 * @brief Scheduler types for GEMM pipeline
 */
enum class SchedulerEnum {
    Intrawave = 0, ///< Intra-wave scheduling
    Interwave = 1  ///< Inter-wave scheduling
};

struct SchedulerEnumInfo
{
    std::string name;
    std::string short_name;
    std::string tag;
};

// Scheduler type to tag mapping
static const std::unordered_map<SchedulerEnum, SchedulerEnumInfo> g_scheduler_info_map{
    {SchedulerEnum::Intrawave, {"Intrawave", "Intrawave", "ck_tile::GemmPipelineScheduler::Intrawave"}},
    {SchedulerEnum::Interwave, {"Interwave", "Interwave", "ck_tile::GemmPipelineScheduler::Interwave"}}
};

/**
 * @enum EpilogueEnum
 * @brief Epilogue types for GEMM
 */
enum class EpilogueEnum {
    Default = 0,   ///< Default epilogue
    Cshuffle = 1   ///< CShuffle epilogue
};

struct EpilogueEnumInfo
{
    std::string name;
    std::string short_name;
    std::string tag;
};

// Default epilogue template string
static const std::string g_default_epilogue_tag = R"(
using GemmEpilogue_{{idx}} = ck_tile::DefaultGemm2DEpilogue<
    ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                          BDataType,
                                          AccDataType,
                                          CDataType,
                                          CLayout,
                                          {{is_pad_m}},
                                          {{is_pad_n}},
                                          {{m_warp_tile}},
                                          {{n_warp_tile}},
                                          {{k_warp_tile}},
                                          {{c_permute}},
                                          true, // UseRawStore_
{% if split_k == 1 %}
                                          ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                          ck_tile::memory_operation_enum::set>{}
{% else %}
                                          ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                          ck_tile::memory_operation_enum::atomic_add>{}
{% endif %}
                                          >>;
)";

// CShuffle epilogue template string
static const std::string g_cshuffle_epilogue_tag = R"(
using GemmEpilogue_{{idx}} = ck_tile::CShuffleEpilogue<
    ck_tile::CShuffleEpilogueProblem<ADataType,
                                     BDataType,
                                     DsDataType,
                                     AccDataType,
                                     CDataType,
                                     DsLayout,
                                     CLayout,
                                     {{elementwise_kind}},
                                     UniversalGemmProblem_{{idx}}::kBlockSize,
                                     TilePartitioner_{{idx}}::MPerBlock,
                                     TilePartitioner_{{idx}}::NPerBlock,
                                     {{m_warp}},
                                     {{n_warp}},
                                     {{m_warp_tile}},
                                     {{n_warp_tile}},
                                     {{k_warp_tile}},
                                     UniversalGemmProblem_{{idx}}::TransposeC,
{% if split_k == 1 %}
                                          ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                          ck_tile::memory_operation_enum::set>{}
{% else %}
                                          ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                          ck_tile::memory_operation_enum::atomic_add>{}
{% endif %}
                                    {{num_wave_groups}},
                                    false, /*FixedVectorSize_*/
                                    1, /*VectorSizeC_*/
                                    >>;
)";


// Epilogue type to tag mapping
static const std::unordered_map<EpilogueEnum, EpilogueEnumInfo> g_epilogue_info_map{
    {EpilogueEnum::Default, {"DefaultGemm2DEpilogue", "Default", g_default_epilogue_tag}},
    {EpilogueEnum::Cshuffle, {"CShuffleEpilogue", "CShuffle", g_cshuffle_epilogue_tag}}
};

/**
 * @enum LayoutType
 * @brief Defines tensor layout types for GEMM operations
 */
enum class LayoutType : int {
    ColumnMajor = 0,  ///< Column-major layout (Fortran-style)
    RowMajor    = 1,  ///< Row-major layout (C-style)
    COUNT             ///< Used for validation
};

struct LayoutTypeInfo
{
    std::string name;
    std::string short_name;
    std::string tag;
};


// Layout type to tag mapping
static const std::unordered_map<LayoutType, LayoutTypeInfo> g_layout_type_info_map{
    {LayoutType::ColumnMajor, {"ColumnMajor", "C", "ck_tile::tensor_layout::gemm::ColumnMajor"}},
    {LayoutType::RowMajor, {"RowMajor", "R", "ck_tile::tensor_layout::gemm::RowMajor"}}
};


// ====================== Utility Functions ======================

/**
 * @brief Gets the name string for a gemm kind
 * @param kind The gemm kind to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetGemmKindName(GemmKind kind) {
    auto it = g_gemm_kind_info_map.find(kind);
    return it != g_gemm_kind_info_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the short name string for a gemm kind
 * @param kind The gemm kind to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetGemmKindShortName(GemmKind kind) {
    auto it = g_gemm_kind_info_map.find(kind);
    return it != g_gemm_kind_info_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the tag string for a gemm kind
 * @param kind The gemm kind to query
 * @return The tag string, or "unknown" if not found
 */
inline std::string GetGemmKindTag(GemmKind kind) {
    auto it = g_gemm_kind_info_map.find(kind);
    return it != g_gemm_kind_info_map.end() ? it->second.tag : "unknown";
}

/**
 * @brief Gets the name string for an elementwise kind
 */
inline std::string GetElementwiseKindName(ElementwiseKind kind) {
    auto it = g_elementwise_kind_info_map.find(kind);
    return it != g_elementwise_kind_info_map.end() ? it->second.name : "unknown";
}
/**
 * @brief Gets the short name string for an elementwise kind
 */
inline std::string GetElementwiseKindShortName(ElementwiseKind kind) {
    auto it = g_elementwise_kind_info_map.find(kind);
    return it != g_elementwise_kind_info_map.end() ? it->second.short_name : "unknown";
}
/**
 * @brief Gets the tag string for an elementwise kind
 */
inline std::string GetElementwiseKindTag(ElementwiseKind kind) {
    auto it = g_elementwise_kind_info_map.find(kind);
    return it != g_elementwise_kind_info_map.end() ? it->second.tag : "unknown";
}

/**
 * @brief Gets the name string for a pipeline enum
 */
inline std::string GetPipelineEnumName(PipelineEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.name : "unknown";
}
/**
 * @brief Gets the short name string for a pipeline enum
 */
inline std::string GetPipelineEnumShortName(PipelineEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the base tag string for a pipeline enum
 */
inline std::string GetPipelineEnumBaseTag(PipelineEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.base_tag : "unknown";
}

/**
 * @brief Gets the main tag string for a pipeline enum
 */
inline std::string GetPipelineEnumMainTag(PipelineEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.main_tag : "unknown";
}

/**
 * @brief Gets the PipelineEnum from a string (name or short_name)
 * @param str The string to match
 * @return The corresponding PipelineEnum, or PipelineEnum::Mem if not found
 */
inline PipelineEnum GetPipelineEnumFromString(const std::string& str)
{
    for (const auto& [e, info] : g_pipeline_info_map) {
        if (info.name == str || info.short_name == str) {
            return e;
        }
    }
    return PipelineEnum::Mem;
}


/**
 * @brief Gets the name string for a scheduler enum
 */
inline std::string GetSchedulerEnumName(SchedulerEnum s) {
    auto it = g_scheduler_info_map.find(s);
    return it != g_scheduler_info_map.end() ? it->second.name : "unknown";
}
/**
 * @brief Gets the short name string for a scheduler enum
 */
inline std::string GetSchedulerEnumShortName(SchedulerEnum s) {
    auto it = g_scheduler_info_map.find(s);
    return it != g_scheduler_info_map.end() ? it->second.short_name : "unknown";
}
/**
 * @brief Gets the tag string for a scheduler enum
 */
inline std::string GetSchedulerEnumTag(SchedulerEnum s) {
    auto it = g_scheduler_info_map.find(s);
    return it != g_scheduler_info_map.end() ? it->second.tag : "unknown";
}

/**
 * @brief Gets the SchedulerEnum from a string (name or short_name)
 * @param str The string to match
 * @return The corresponding SchedulerEnum, or SchedulerEnum::Intrawave if not found
 */
inline SchedulerEnum GetSchedulerEnumFromString(const std::string& str)
{
    for (const auto& [e, info] : g_scheduler_info_map) {
        if (info.name == str || info.short_name == str) {
            return e;
        }
    }
    return SchedulerEnum::Intrawave;
}


/**
 * @brief Gets the name string for an epilogue enum
 */
inline std::string GetEpilogueEnumName(EpilogueEnum e) {
    auto it = g_epilogue_info_map.find(e);
    return it != g_epilogue_info_map.end() ? it->second.name : "unknown";
}
/**
 * @brief Gets the short name string for an epilogue enum
 */
inline std::string GetEpilogueEnumShortName(EpilogueEnum e) {
    auto it = g_epilogue_info_map.find(e);
    return it != g_epilogue_info_map.end() ? it->second.short_name : "unknown";
}
/**
 * @brief Gets the tag string for an epilogue enum (code template)
 */
inline std::string GetEpilogueEnumTag(EpilogueEnum e) {
    auto it = g_epilogue_info_map.find(e);
    return it != g_epilogue_info_map.end() ? it->second.tag : "unknown";
}

/**
 * @brief Gets the EpilogueEnum from a string (name or short_name)
 * @param str The string to match
 * @return The corresponding EpilogueEnum, or EpilogueEnum::Default if not found
 */
inline EpilogueEnum GetEpilogueEnumFromString(const std::string& str)
{
    for (const auto& [e, info] : g_epilogue_info_map) {
        if (info.name == str || info.short_name == str) {
            return e;
        }
    }
    return EpilogueEnum::Default;
}

/**
 * @brief Gets the name string for a layout type
 */
inline std::string GetLayoutTypeName(LayoutType l) {
    auto it = g_layout_type_info_map.find(l);
    return it != g_layout_type_info_map.end() ? it->second.name : "unknown";
}
/**
 * @brief Gets the short name string for a layout type
 */
inline std::string GetLayoutTypeShortName(LayoutType l) {
    auto it = g_layout_type_info_map.find(l);
    return it != g_layout_type_info_map.end() ? it->second.short_name : "unknown";
}
/**
 * @brief Gets the tag string for a layout type
 */
inline std::string GetLayoutTypeTag(LayoutType l) {
    auto it = g_layout_type_info_map.find(l);
    return it != g_layout_type_info_map.end() ? it->second.tag : "unknown";
}


} // namespace flashck