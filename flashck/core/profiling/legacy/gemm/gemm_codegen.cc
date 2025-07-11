#include "flashck/core/profiling/codegen/legacy/gemm/gemm_codegen.h"

namespace flashck {

std::string GemmTileDesc::GetConfigName()
{
    return Sprintf("{block_size}_{m_per_block}_{}_{}_{}_{}_{}_{}_{}_{}",
                   block_size_,
                   m_per_block_,
                   n_per_block_,
                   k_per_block_,
                   ak1_,
                   bk1_,
                   m_per_xdl_,
                   n_per_xdl_,
                   m_xdl_per_wave_,
                   n_xdl_per_wave_);
}

std::string GemmTileDesc::Emit()
{

    std::string source = R"(
{% for value in param %}
{% if value!=0 %}
    {{value}},
{% endif %}
{% endfor %}
)";

    jinja2::ValuesMap value_map{{"param",
                                 jinja2::ValuesList{block_size_,
                                                    m_per_block_,
                                                    n_per_block_,
                                                    k_per_block_,
                                                    ak1_,
                                                    bk1_,
                                                    m_per_xdl_,
                                                    n_per_xdl_,
                                                    m_xdl_per_wave_,
                                                    n_xdl_per_wave_}}};

    return TemplateLoadAndRender(source, value_map);
}

AttnTileDesc::AttnTileDesc(int64_t block_size,
                           int64_t m_per_block,
                           int64_t n_per_block,
                           int64_t k_per_block,
                           int64_t gemm1_n_per_block,
                           int64_t gemm1_k_per_block,
                           int64_t ak1,
                           int64_t bk1,
                           int64_t b1k1,
                           int64_t m_per_xdl,
                           int64_t n_per_xdl,
                           int64_t m_xdl_per_wave,
                           int64_t n_xdl_per_wave,
                           int64_t gemm1_n_xdl_per_wave):
    block_size_(block_size),
    m_per_block_(m_per_block),
    n_per_block_(n_per_block),
    k_per_block_(k_per_block),
    gemm1_n_per_block_(gemm1_n_per_block),
    gemm1_k_per_block_(gemm1_k_per_block),
    ak1_(ak1),
    bk1_(bk1),
    b1k1_(b1k1),
    m_per_xdl_(m_per_xdl),
    n_per_xdl_(n_per_xdl),
    m_xdl_per_wave_(m_xdl_per_wave),
    n_xdl_per_wave_(n_xdl_per_wave),
    gemm1_n_xdl_per_wave_(gemm1_n_xdl_per_wave)
{
}

std::string AttnTileDesc::GetConfigName()
{
    std::string config_name = Sprintf("{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                                      block_size_,
                                      m_per_block_,
                                      n_per_block_,
                                      k_per_block_,
                                      gemm1_n_per_block_,
                                      gemm1_k_per_block_,
                                      ak1_,
                                      bk1_,
                                      b1k1_,
                                      m_per_xdl_,
                                      n_per_xdl_,
                                      m_xdl_per_wave_,
                                      n_xdl_per_wave_,
                                      gemm1_n_xdl_per_wave_);

    return config_name;
}

std::string AttnTileDesc::Emit()
{
    std::string source = R"(
{% for value in param %}
{% if value!=0 %}
    {{value}},
{% endif %}
{% endfor %}
)";

    jinja2::ValuesMap value_map{{"param",
                                 jinja2::ValuesList{block_size_,
                                                    m_per_block_,
                                                    n_per_block_,
                                                    k_per_block_,
                                                    gemm1_n_per_block_,
                                                    gemm1_k_per_block_,
                                                    ak1_,
                                                    bk1_,
                                                    b1k1_,
                                                    m_per_xdl_,
                                                    n_per_xdl_,
                                                    m_xdl_per_wave_,
                                                    n_xdl_per_wave_,
                                                    gemm1_n_xdl_per_wave_}}};

    return TemplateLoadAndRender(source, value_map);
}

BlockTransferDesc::BlockTransferDesc(std::vector<int64_t> thread_cluster_length,
                                     std::vector<int64_t> thread_cluster_arrange_order,
                                     std::vector<int64_t> src_access_order,
                                     int64_t              src_vector_dim,
                                     int64_t              src_scalar_per_vector,
                                     int64_t              dst_scalar_per_vector,
                                     int64_t              add_extra_dim,
                                     bool                 add_extra_dim_flag):
    thread_cluster_length_(thread_cluster_length),
    thread_cluster_arrange_order_(thread_cluster_arrange_order),
    src_access_order_(src_access_order),
    src_vector_dim_(src_vector_dim),
    src_scalar_per_vector_(src_scalar_per_vector),
    dst_scalar_per_vector_(dst_scalar_per_vector),
    add_extra_dim_(add_extra_dim),
    add_extra_dim_flag_(add_extra_dim_flag)
{

    thread_cluster_length_vec_.reserve(thread_cluster_length_.size());
    thread_cluster_arrange_order_vec_.reserve(thread_cluster_arrange_order_.size());
    src_access_order_vec_.reserve(src_access_order_.size());

    std::transform(thread_cluster_length_.begin(),
                   thread_cluster_length_.end(),
                   std::back_inserter(thread_cluster_length_vec_),
                   [](const int64_t num) { return std::to_string(num); });

    std::transform(thread_cluster_arrange_order_.begin(),
                   thread_cluster_arrange_order_.end(),
                   std::back_inserter(thread_cluster_arrange_order_vec_),
                   [](const int64_t num) { return std::to_string(num); });

    std::transform(src_access_order_.begin(),
                   src_access_order_.end(),
                   std::back_inserter(src_access_order_vec_),
                   [](const int64_t num) { return std::to_string(num); });
}

std::string BlockTransferDesc::GetConfigName()
{

    jinja2::ValuesMap value_map{{{"thread_cluster_length", jinja2::Reflect(thread_cluster_length_vec_)},
                                 {"thread_cluster_arrange_order", jinja2::Reflect(thread_cluster_arrange_order_vec_)},
                                 {"src_access_order", jinja2::Reflect(src_access_order_vec_)},
                                 {"src_vector_dim", src_vector_dim_},
                                 {"src_scalar_per_vector", src_scalar_per_vector_},
                                 {"dst_scalar_per_vector", dst_scalar_per_vector_},
                                 {"add_extra_dim", add_extra_dim_},
                                 {"add_extra_dim_flag", add_extra_dim_flag_}}};

    std::string source = R"(
    S{{thread_cluster_length|join('_')}}S
    _S{{thread_cluster_arrange_order|join('_')}}S
    _S{{src_access_order|join('_')}}S
    _{{src_vector_dim}}
    _{{src_scalar_per_vector}}
    _{{dst_scalar_per_vector}}
    _{{add_extra_dim}}

)";

    return TemplateLoadAndRender(source, value_map);
}

std::string BlockTransferDesc::Emit()
{
    jinja2::ValuesMap value_map{{{"thread_cluster_length", jinja2::Reflect(thread_cluster_length_vec_)},
                                 {"thread_cluster_arrange_order", jinja2::Reflect(thread_cluster_arrange_order_vec_)},
                                 {"src_access_order", jinja2::Reflect(src_access_order_vec_)},
                                 {"src_vector_dim", src_vector_dim_},
                                 {"src_scalar_per_vector", src_scalar_per_vector_},
                                 {"dst_scalar_per_vector", dst_scalar_per_vector_},
                                 {"add_extra_dim", add_extra_dim_},
                                 {"add_extra_dim_flag", add_extra_dim_flag_}}};

    std::string source = R"(
    ck::Sequence<{{thread_cluster_length|join(',')}}>, // thread_cluster_length
    ck::Sequence<{{thread_cluster_arrange_order|join(',')}}>, // thread_cluster_arrange_order
    ck::Sequence<{{src_access_order|join(',')}}>, // src_access_order
    {{src_vector_dim}}, // src_vector_dim
    {{src_scalar_per_vector}}, // src_scalar_per_vector
    {{dst_scalar_per_vector}}, // dst_scalar_per_vector
{% if add_extra_dim_flag %}
    {% if add_extra_dim==1 %}true, {% else %}false,{% endif %} //add_extra_dim
{% else %}
    {{add_extra_dim}}, // add_extra_dim
{% endif %}
)";

    return TemplateLoadAndRender(source, value_map);
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
    m_n_block_wave_per_xdl_vec_.reserve(m_n_block_wave_per_xdl_.size());

    std::transform(m_n_block_wave_per_xdl_.begin(),
                   m_n_block_wave_per_xdl_.end(),
                   std::back_inserter(m_n_block_wave_per_xdl_vec_),
                   [](const int64_t num) { return std::to_string(num); });
}

std::string CBlockTransferDesc::GetConfigName()
{

    jinja2::ValuesMap value_map{{
        {"m_xdl_per_wave", m_xdl_per_wave_},
        {"n_xdl_per_wave", n_xdl_per_wave_},
        {"m_n_block_wave_per_xdl", jinja2::Reflect(m_n_block_wave_per_xdl_vec_)},
        {"scalar_per_vector", scalar_per_vector_},
    }};

    std::string source = R"(
    {{m_xdl_per_wave}}
    _{{n_xdl_per_wave}}
    {{m_n_block_wave_per_xdl|join('_')}}S
    {{scalar_per_vector}}
)";

    return TemplateLoadAndRender(source, value_map);
}

std::string CBlockTransferDesc::Emit()
{

    jinja2::ValuesMap value_map{{
        {"m_xdl_per_wave", m_xdl_per_wave_},
        {"n_xdl_per_wave", n_xdl_per_wave_},
        {"m_n_block_wave_per_xdl", jinja2::Reflect(m_n_block_wave_per_xdl_vec_)},
        {"scalar_per_vector", scalar_per_vector_},
    }};

    std::string source = R"(
    {{m_xdl_per_wave}}, // m_xdl_per_wave
    {{n_xdl_per_wave}}, // n_xdl_per_wave
    ck::Sequence<{{m_n_block_wave_per_xdl|join(',')}}>, // m_n_block_wave_per_xdl
    {{scalar_per_vector}} // scalar_per_vector
)";

    return TemplateLoadAndRender(source, value_map);
}

std::string GemmOperation::GetConfigName()
{
    std::string io_name = Sprintf("{}_{}{}{}{}_{}{}{}_{}",
                                  g_gemm_operation_kind_names.find(kind_)->second,
                                  DataTypeToShortString(a_tensor_desc_.element_),
                                  DataTypeToShortString(b_tensor_desc_.element_),
                                  DataTypeToShortString(c_tensor_desc_.element_),
                                  DataTypeToShortString(accumulator_type_),
                                  g_short_layout_names.find(a_tensor_desc_.layout_)->second,
                                  g_short_layout_names.find(b_tensor_desc_.layout_)->second,
                                  g_short_layout_names.find(c_tensor_desc_.layout_)->second,
                                  g_short_gemm_spec_names.find(gemm_specialization_)->second);

    std::string extra_tile = "";

    if (c_block_transfer_.has_value()) {
        if (c_block_transfer_->scalar_per_vector_ == 4) {
            extra_tile = "_C4";
        }
        else if (c_block_transfer_->scalar_per_vector_ == 1) {
            extra_tile = "_C1";
        }
    }
    else if (mask_c_block_transfer_.has_value()) {
        if (mask_c_block_transfer_->scalar_per_vector_ == 4) {
            extra_tile = "_C4";
        }
        else if (mask_c_block_transfer_->scalar_per_vector_ == 1) {
            extra_tile = "_C1";
        }
    }

    std::string tile_name = kind_ == GemmOperationKind::BatchGemmSoftmaxGemmPermute ?
                                attn_tile_desc_.GetConfigName() + extra_tile :
                                tile_desc_.GetConfigName() + extra_tile;

    std::string casual_name =
        kind_ == GemmOperationKind::BatchGemmSoftmaxGemmPermute ?
            g_short_tensor_operation_names_map.find(mask_c_block_transfer_->causal_mask_)->second :
            "";

    std::string config_name = Sprintf(
        "{}_{}_{}", io_name, tile_name, g_short_tensor_operation_names_map.find(c_element_op_)->second + casual_name);

    return config_name;
}

std::string GemmOperation::Emit()
{

    std::string source;

    jinja2::ValuesMap value_map{{"name", name_value},
                                {"gemm_kind", g_gemm_operation_kind_names.find(kind_)->second},
                                {"kernel_type", g_kernel_tag.find(kernel_type_)->second},
                                {"kernel_type_value", static_cast<int>(kernel_type_)},
                                {"ALayout", g_layout_tag.find(a_tensor_desc_.layout_)->second},
                                {"BLayout", g_layout_tag.find(b_tensor_desc_.layout_)->second},
                                {"CLayout", g_layout_tag.find(c_tensor_desc_.layout_)->second},
                                {"ADType", DataTypeToString(a_tensor_desc_.element_)},
                                {"BDType", DataTypeToString(b_tensor_desc_.element_)},
                                {"CDType", DataTypeToString(c_tensor_desc_.element_)},
                                {"AccDType", DataTypeToString(accumulator_type_)},
                                {"CShuffleDType", DataTypeToString(c_tensor_desc_.element_)},
                                {"A_elem_op", g_tensor_operation_tag.find(a_element_op_)->second},
                                {"B_elem_op", g_tensor_operation_tag.find(b_element_op_)->second},
                                {"C_elem_op", g_tensor_operation_tag.find(c_element_op_)->second},
                                {"GemmSpecialization", g_gemm_specialization_tag.find(gemm_specialization_)->second},
                                {"tile_config", tile_config_value},
                                {"a_block_transfer", a_block_transfer_value},
                                {"b_block_transfer", b_block_transfer_value},
                                {"b1_block_transfer", b1_block_transfer_value},
                                {"c_block_transfer", c_block_transfer_value},
                                {"DsDType", ds_dtype_value},
                                {"DsLayout", ds_layout_value},
                                {"EDType", e_dtype_value}};

    return TemplateLoadAndRender(source, value_map);
}

}  // namespace flashck
