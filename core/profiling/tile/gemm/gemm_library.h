#pragma once

#include <string>
#include <unordered_map>

namespace flashck {

namespace tile{

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
 * @enum PipelineVersionEnum
 * @brief Pipeline types for GEMM kernel
 */
enum class PipelineVersionEnum {
    Mem = 0,
    Compute_V3 = 1,
    Compute_V4 = 2,
    Compute_V5 = 3,
    Preshuffle = 4
};


struct PipelineVersionEnumInfo
{
    std::string name;
    std::string short_name;
    std::string base_tag;
    std::string main_tag;
};

// Pipeline type to tag mapping
static const std::unordered_map<PipelineVersionEnum, PipelineVersionEnumInfo> g_pipeline_info_map{
    {PipelineVersionEnum::Mem, {"GemmPipelineAgBgCrMem", "Mem", "ck_tile::BaseGemmPipelineAgBgCrMem", "ck_tile::GemmPipelineAgBgCrMem"}},
    {PipelineVersionEnum::Compute_V3, {"GemmPipelineAgBgCrCompV3", "CompV3", "ck_tile::BaseGemmPipelineAgBgCrCompV3", "ck_tile::GemmPipelineAgBgCrCompV3"}},
    {PipelineVersionEnum::Compute_V4, {"GemmPipelineAgBgCrCompV4", "CompV4", "ck_tile::BaseGemmPipelineAgBgCrCompV4", "ck_tile::GemmPipelineAgBgCrCompV4"}},
    {PipelineVersionEnum::Compute_V5, {"GemmPipelineAgBgCrCompV5", "CompV5", "ck_tile::BaseGemmPipelineAgBgCrCompV5", "ck_tile::GemmPipelineAgBgCrCompV5"}},
    {PipelineVersionEnum::Preshuffle, {"WeightPreshufflePipelineAGmemBGmemCRegV1", "Preshuffle", "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV1", "ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV1"}}
};

/**
 * @enum PipelineSchedulerEnum
 * @brief Scheduler types for GEMM pipeline
 */
enum class PipelineSchedulerEnum {
    Intrawave = 0, ///< Intra-wave scheduling
    Interwave = 1  ///< Inter-wave scheduling
};

struct PipelineSchedulerEnumInfo
{
    std::string name;
    std::string short_name;
    std::string tag;
};

// Scheduler type to tag mapping
static const std::unordered_map<PipelineSchedulerEnum, PipelineSchedulerEnumInfo> g_scheduler_info_map{
    {PipelineSchedulerEnum::Intrawave, {"Intrawave", "Intrawave", "ck_tile::GemmPipelineScheduler::Intrawave"}},
    {PipelineSchedulerEnum::Interwave, {"Interwave", "Interwave", "ck_tile::GemmPipelineScheduler::Interwave"}}
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
inline std::string GetPipelineVersionEnumName(PipelineVersionEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.name : "unknown";
}
/**
 * @brief Gets the short name string for a pipeline enum
 */
inline std::string GetPipelineVersionEnumShortName(PipelineVersionEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the base tag string for a pipeline enum
 */
inline std::string GetPipelineVersionEnumBaseTag(PipelineVersionEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.base_tag : "unknown";
}

/**
 * @brief Gets the main tag string for a pipeline enum
 */
inline std::string GetPipelineVersionEnumMainTag(PipelineVersionEnum p) {
    auto it = g_pipeline_info_map.find(p);
    return it != g_pipeline_info_map.end() ? it->second.main_tag : "unknown";
}

/**
 * @brief Gets the PipelineVersionEnum from a string (name or short_name)
 * @param str The string to match
 * @return The corresponding PipelineVersionEnum, or PipelineVersionEnum::Mem if not found
 */
inline PipelineVersionEnum GetPipelineVersionEnumFromString(const std::string& str)
{
    for (const auto& [e, info] : g_pipeline_info_map) {
        if (info.name == str || info.short_name == str) {
            return e;
        }
    }
    return PipelineVersionEnum::Mem;
}


/**
 * @brief Gets the name string for a scheduler enum
 */
inline std::string GetPipelineSchedulerEnumName(PipelineSchedulerEnum s) {
    auto it = g_scheduler_info_map.find(s);
    return it != g_scheduler_info_map.end() ? it->second.name : "unknown";
}
/**
 * @brief Gets the short name string for a scheduler enum
 */
inline std::string GetPipelineSchedulerEnumShortName(PipelineSchedulerEnum s) {
    auto it = g_scheduler_info_map.find(s);
    return it != g_scheduler_info_map.end() ? it->second.short_name : "unknown";
}
/**
 * @brief Gets the tag string for a scheduler enum
 */
inline std::string GetPipelineSchedulerEnumTag(PipelineSchedulerEnum s) {
    auto it = g_scheduler_info_map.find(s);
    return it != g_scheduler_info_map.end() ? it->second.tag : "unknown";
}

/**
 * @brief Gets the PipelineSchedulerEnum from a string (name or short_name)
 * @param str The string to match
 * @return The corresponding PipelineSchedulerEnum, or PipelineSchedulerEnum::Intrawave if not found
 */
inline PipelineSchedulerEnum GetPipelineSchedulerEnumFromString(const std::string& str)
{
    for (const auto& [e, info] : g_scheduler_info_map) {
        if (info.name == str || info.short_name == str) {
            return e;
        }
    }
    return PipelineSchedulerEnum::Intrawave;
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

} // namespace tile

} // namespace flashck