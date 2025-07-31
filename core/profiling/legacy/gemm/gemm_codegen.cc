#include "core/profiling/legacy/gemm/gemm_codegen.h"

#include <algorithm>
#include <string>

#include "core/utils/macros.h"
#include "core/utils/string_utils.h"

namespace flashck {

namespace legacy {

std::string GemmTileDesc::GetInstanceName() const
{
    return Sprintf(
        "{scale_block_size}_{block_size}_{m_per_block}_{n_per_block}_{k_per_block}_{ak1}_{bk1}_{m_per_xdl}_{n_per_xdl}_{m_xdl_per_wave}_{n_xdl_per_wave}",
        fmt::arg("scale_block_size", scale_block_size_),
        fmt::arg("block_size", block_size_),
        fmt::arg("m_per_block", m_per_block_),
        fmt::arg("n_per_block", n_per_block_),
        fmt::arg("k_per_block", k_per_block_),
        fmt::arg("ak1", a_k1_),
        fmt::arg("bk1", b_k1_),
        fmt::arg("m_per_xdl", m_per_xdl_),
        fmt::arg("n_per_xdl", n_per_xdl_),
        fmt::arg("m_xdl_per_wave", m_xdl_per_wave_),
        fmt::arg("n_xdl_per_wave", n_xdl_per_wave_));
}

std::string GemmTileDesc::Emit() const
{
    std::string tpl = R"(
    {{scale_block_size}}, // scale_block_size
    {{block_size}}, // block_size
    {{m_per_block}}, // m_per_block
    {{n_per_block}}, // n_per_block
    {{k_per_block}}, // k_per_block
    {{a_k1}}, // a_k1
    {{b_k1}}, // b_k1
    {{m_per_xdl}}, // m_per_xdl
    {{n_per_xdl}}, // n_per_xdl
    {{m_xdl_per_wave}}, // m_xdl_per_wave
    {{n_xdl_per_wave}} // n_xdl_per_wave
)";

    jinja2::ValuesMap values{{"block_size", block_size_},
                             {"m_per_block", m_per_block_},
                             {"n_per_block", n_per_block_},
                             {"k_per_block", k_per_block_},
                             {"a_k1", a_k1_},
                             {"b_k1", b_k1_},
                             {"m_per_xdl", m_per_xdl_},
                             {"n_per_xdl", n_per_xdl_},
                             {"m_xdl_per_wave", m_xdl_per_wave_},
                             {"n_xdl_per_wave", n_xdl_per_wave_}};

    return TEMPLATE_CHECK(tpl, values, "GemmTileDesc::Emit");
}

BlockTransferDesc::BlockTransferDesc(const std::vector<int64_t>& thread_cluster_length,
                                     const std::vector<int64_t>& thread_cluster_arrange_order,
                                     const std::vector<int64_t>& src_access_order,
                                     int64_t                     src_vector_dim,
                                     int64_t                     src_scalar_per_vector,
                                     int64_t                     dst_scalar_per_vector,
                                     int64_t                     add_extra_dim):
    thread_cluster_length_(thread_cluster_length),
    thread_cluster_arrange_order_(thread_cluster_arrange_order),
    src_access_order_(src_access_order),
    src_vector_dim_(src_vector_dim),
    src_scalar_per_vector_(src_scalar_per_vector),
    dst_scalar_per_vector_(dst_scalar_per_vector),
    add_extra_dim_(add_extra_dim)
{
}

std::string BlockTransferDesc::GetInstanceName() const
{

    jinja2::ValuesMap value_map{{{"thread_cluster_length", jinja2::Reflect(thread_cluster_length_)},
                                 {"thread_cluster_arrange_order", jinja2::Reflect(thread_cluster_arrange_order_)},
                                 {"src_access_order", jinja2::Reflect(src_access_order_)},
                                 {"src_vector_dim", src_vector_dim_},
                                 {"src_scalar_per_vector", src_scalar_per_vector_},
                                 {"dst_scalar_per_vector", dst_scalar_per_vector_},
                                 {"add_extra_dim", add_extra_dim_}}};

    std::string tpl = R"(
    S{{thread_cluster_length|join('_')}}S
    _S{{thread_cluster_arrange_order|join('_')}}S
    _S{{src_access_order|join('_')}}S
    _{{src_vector_dim}}
    _{{src_scalar_per_vector}}
    _{{dst_scalar_per_vector}}
    _{{add_extra_dim}}

)";
    return TEMPLATE_CHECK(tpl, value_map, "BlockTransferDesc::GetInstanceName");
}

std::string BlockTransferDesc::Emit() const
{
    jinja2::ValuesMap value_map{{{"thread_cluster_length", jinja2::Reflect(thread_cluster_length_vec_)},
                                 {"thread_cluster_arrange_order", jinja2::Reflect(thread_cluster_arrange_order_vec_)},
                                 {"src_access_order", jinja2::Reflect(src_access_order_vec_)},
                                 {"src_vector_dim", src_vector_dim_},
                                 {"src_scalar_per_vector", src_scalar_per_vector_},
                                 {"dst_scalar_per_vector", dst_scalar_per_vector_},
                                 {"add_extra_dim", add_extra_dim_}}};

    std::string tpl = R"(
    ck::Sequence<{{thread_cluster_length|join(',')}}>, // thread_cluster_length
    ck::Sequence<{{thread_cluster_arrange_order|join(',')}}>, // thread_cluster_arrange_order
    ck::Sequence<{{src_access_order|join(',')}}>, // src_access_order
    {{src_vector_dim}}, // src_vector_dim
    {{src_scalar_per_vector}}, // src_scalar_per_vector
    {{dst_scalar_per_vector}}, // dst_scalar_per_vector
    {{add_extra_dim}}, // add_extra_dim
)";

    return TEMPLATE_CHECK(tpl, value_map, "BlockTransferDesc::Emit");
}

CBlockTransferDesc::CBlockTransferDesc(int64_t              m_xdl_per_wave,
                                       int64_t              n_xdl_per_wave,
                                       std::vector<int64_t> m_n_block_wave_per_xdl,
                                       int64_t              scalar_per_vector):
    m_xdl_per_wave_(m_xdl_per_wave),
    n_xdl_per_wave_(n_xdl_per_wave),
    m_n_block_wave_per_xdl_(m_n_block_wave_per_xdl),
    scalar_per_vector_(scalar_per_vector)
{
}

std::string CBlockTransferDesc::GetInstanceName() const
{
    jinja2::ValuesMap value_map{{
        {"m_xdl_per_wave", m_xdl_per_wave_},
        {"n_xdl_per_wave", n_xdl_per_wave_},
        {"m_n_block_wave_per_xdl", jinja2::Reflect(m_n_block_wave_per_xdl_)},
        {"scalar_per_vector", scalar_per_vector_},
    }};

    std::string tpl = R"(
    {{m_xdl_per_wave}}
    _{{n_xdl_per_wave}}
    _S{{m_n_block_wave_per_xdl|join('_')}}S
    _{{scalar_per_vector}}
)";

    return TEMPLATE_CHECK(tpl, value_map, "CBlockTransferDesc::GetInstanceName");
}

std::string CBlockTransferDesc::Emit() const
{
    jinja2::ValuesMap value_map{{
        {"m_xdl_per_wave", m_xdl_per_wave_},
        {"n_xdl_per_wave", n_xdl_per_wave_},
        {"m_n_block_wave_per_xdl", jinja2::Reflect(m_n_block_wave_per_xdl_)},
        {"scalar_per_vector", scalar_per_vector_},
    }};

    std::string tpl = R"(
    {{m_xdl_per_wave}}, // m_xdl_per_wave
    {{n_xdl_per_wave}}, // n_xdl_per_wave
    ck::Sequence<{{m_n_block_wave_per_xdl|join(',')}}>, // m_n_block_wave_per_xdl
    {{scalar_per_vector}} // scalar_per_vector
)";

    return TEMPLATE_CHECK(tpl, value_map, "CBlockTransferDesc::Emit");
}

std::string GemmCodeGen::GetInstanceName() const
{
    return Sprintf(
        "{kind}_{epilogue}_{tile_desc}_{a_layout}{b_layout}{c_layout}_{a_dtype}{b_dtype}{c_dtype}_{a_block_transfer}_{b_block_transfer}_{c_block_transfer}_{pipeline_scheduler}_{pipeline_version}_{gemm_spec}",
        fmt::arg("kind", GetGemmKindShortName(problem_.kind_)),
        fmt::arg("epilogue", GetEpilogueShortName(problem_.epilogue_)),
        fmt::arg("tile_desc", tile_desc_.GetInstanceName()),
        fmt::arg("a_layout", GetLayoutShortName(problem_.a_layout_)),
        fmt::arg("b_layout", GetLayoutShortName(problem_.b_layout_)),
        fmt::arg("c_layout", GetLayoutShortName(problem_.c_layout_)),
        fmt::arg("a_dtype", DataTypeToString(problem_.a_dtype_)),
        fmt::arg("b_dtype", DataTypeToString(problem_.b_dtype_)),
        fmt::arg("c_dtype", DataTypeToString(problem_.c_dtype_)),
        fmt::arg("a_block_transfer", a_block_desc_.GetInstanceName()),
        fmt::arg("b_block_transfer", b_block_desc_.GetInstanceName()),
        fmt::arg("c_block_transfer", c_block_desc_.GetInstanceName()),
        fmt::arg("pipeline_scheduler", GetSchedulerShortName(pipeline_scheduler_)),
        fmt::arg("pipeline_version", GetPipelineVersionShortName(pipeline_version_)),
        fmt::arg("gemm_spec", GetGemmSpecializationShortName(gemm_spec_)));
}

std::string GemmCodeGen::Emit() const
{
    std::string tpl            = R"(
using GemmInstance_{{idx}} = {{device_tag}}<
    {{a_layout_tag}},  // ALayout
    {{b_layout_tag}},  // BLayout
{% if kind == "gemm_multiple_d" %}
    {{ds_layout_tag}},  // DsLayout (if applicable)
{% endif %}
    {{c_layout_tag}},  // CLayout
{% if kind == "gemm" %}
    {{a_dtype}},       // ADataType
    {{b_dtype}},       // BDataType
    {{c_dtype}},       // CDataType
    {{a_dtype}},       // AccDataType (using same as A for now)
    {{c_dtype}},       // CShuffleDataType (using same as C)
{% elif kind == "gemm_multiple_d" %}
    {{a_dtype}},       // ADataType
    {{b_dtype}},       // BDataType
    {{c_dtype}},       // AccDataType (using same as C)
    {{c_dtype}},       // CShuffleDataType (using same as C for now)
    {{ds_dtype}}       // DsDataType (if applicable)
    {{c_dtype}},       // CDataType
{% endif %}
    ck::tensor_operation::element_wise::PassThrough,  // AElementOp
    ck::tensor_operation::element_wise::PassThrough,  // BElementOp
    {{c_element_op}},  // CElementOp
    {{gemm_spec}},      // GemmSpec
{% if kind == "gemm_multiple_d" %}
    1,                  // NumGemmKPrefetchStage
{% endif %}
    {{tile_desc}},     // TileDesc
    {{a_block_transfer}},  // ABlockTransfer
    {{b_block_transfer}},  // BBlockTransfer
    {{c_block_transfer}},  // CBlockTransfer
    {{pipeline_scheduler}},  // PipelineScheduler
    {{pipeline_version}}, // PipelineVersion
{% if kind == "gemm" %}
    {{c_dtype}}, // compute_type_a(using same as C)
    {{c_dtype}}, // compute_type_b(using same as C)
    false, // permute_a
    false  // permute_b
{% elif kind == "gemm_multiple_d" %}
    {{c_dtype}}, // ComputeDataType(using same as C)
{% endif %}
>;
)";
    auto join_if_not_empty = [](const auto& vec, auto&& func) -> std::string {
        if (vec.empty()) return "";
        std::vector<std::string> tmp;
        tmp.reserve(vec.size());
        std::transform(vec.begin(), vec.end(), std::back_inserter(tmp), func);
        return JoinStrings(tmp, ",");
    };
    std::string ds_dtype_value = join_if_not_empty(problem_.ds_dtype_, DataTypeToString);
    std::string ds_layout_value = join_if_not_empty(problem_.ds_layout_, GetLayoutClassTag);

    static int idx = 0;

    jinja2::ValuesMap value_map{
        {"idx", idx++},
        {"kind", GetGemmKindName(problem_.kind_)},
        {"device_tag", GetGemmKindDeviceTag(problem_.kind_)},
        {"a_layout_tag", GetLayoutClassTag(problem_.a_layout_)},
        {"b_layout_tag", GetLayoutClassTag(problem_.b_layout_)},
        {"c_layout_tag", GetLayoutClassTag(problem_.c_layout_)},
        {"ds_layout_tag", ds_layout_value},
        {"a_dtype", DataTypeToString(problem_.a_dtype_)},
        {"b_dtype", DataTypeToString(problem_.b_dtype_)},
        {"c_dtype", DataTypeToString(problem_.c_dtype_)},
        {"ds_dtype", ds_dtype_value},
        {"tile_desc", tile_desc_.Emit()},
        {"a_block_transfer", a_block_desc_.Emit()},
        {"b_block_transfer", b_block_desc_.Emit()},
        {"c_block_transfer", c_block_desc_.Emit()},
        {"c_element_op", GetEpilogueClassTag(problem_.epilogue_)},
        {"gemm_spec", GetGemmSpecializationClassTag(gemm_spec_)},
        {"pipeline_scheduler", GetSchedulerClassTag(pipeline_scheduler_)},
        {"pipeline_version", GetPipelineVersionTag(pipeline_version_)},
    };

    return TEMPLATE_CHECK(tpl, value_map, "GemmCodeGen::Emit");
}

} // namespace legacy
}  // namespace flashck
