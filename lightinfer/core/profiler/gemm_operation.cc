#include "lightinfer/core/profiler/gemm_operation.h"

#include "lightinfer/core/utils/jinjia2_utils.h"
#include "lightinfer/core/utils/printf.h"

namespace lightinfer {

GemmSpecialization GetGemmSpec(const int64_t m,
                               const int64_t n,
                               const int64_t k,
                               const int64_t m_per_block,
                               const int64_t n_per_block,
                               const int64_t k_per_block)
{

    std::string spec = "";
    if (IntegerDivideCeil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if (IntegerDivideCeil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if (IntegerDivideCeil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";

    return g_gemm_spec_names.at(spec);
}

GemmTileDesc::GemmTileDesc(int64_t block_size,
                           int64_t m_per_block,
                           int64_t n_per_block,
                           int64_t k_per_block,
                           int64_t ak1,
                           int64_t bk1,
                           int64_t m_per_xdl,
                           int64_t n_per_xdl,
                           int64_t m_xdl_per_wave,
                           int64_t n_xdl_per_wave):
    block_size_(block_size),
    m_per_block_(m_per_block),
    n_per_block_(n_per_block),
    k_per_block_(k_per_block),
    ak1_(ak1),
    bk1_(bk1),
    m_per_xdl_(m_per_xdl),
    n_per_xdl_(n_per_xdl),
    m_xdl_per_wave_(m_xdl_per_wave),
    n_xdl_per_wave_(n_xdl_per_wave)
{
}

std::string GemmTileDesc::GetConfigName()
{

    return Sprintf("{}_{}_{}_{}_{}_{}_{}_{}_{}_{}",
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

MaskedCBlockTransferDesc::MaskedCBlockTransferDesc(int64_t              m_xdl_per_wave,
                                                   int64_t              n_xdl_per_wave,
                                                   std::vector<int64_t> m_n_block_wave_per_xdl,
                                                   int64_t              scalar_per_vector,
                                                   TensorOperation      causal_mask):
    m_xdl_per_wave_(m_xdl_per_wave),
    n_xdl_per_wave_(n_xdl_per_wave),
    m_n_block_wave_per_xdl_(m_n_block_wave_per_xdl),
    scalar_per_vector_(scalar_per_vector),
    causal_mask_(causal_mask)
{
    m_n_block_wave_per_xdl_vec_.reserve(m_n_block_wave_per_xdl_.size());

    std::transform(m_n_block_wave_per_xdl_.begin(),
                   m_n_block_wave_per_xdl_.end(),
                   std::back_inserter(m_n_block_wave_per_xdl_vec_),
                   [](const int64_t num) { return std::to_string(num); });
}

std::string MaskedCBlockTransferDesc::GetConfigName()
{
    jinja2::ValuesMap value_map{{{"m_xdl_per_wave", m_xdl_per_wave_},
                                 {"n_xdl_per_wave", n_xdl_per_wave_},
                                 {"m_n_block_wave_per_xdl", jinja2::Reflect(m_n_block_wave_per_xdl_vec_)},
                                 {"scalar_per_vector", scalar_per_vector_},
                                 {"masking_specialization", g_tensor_operation_tag.find(causal_mask_)->second}}};

    std::string source = R"(
    {{m_xdl_per_wave}}
    _{{n_xdl_per_wave}}
    {{m_n_block_wave_per_xdl|join('_')}}S
    {{scalar_per_vector}}
    {{masking_specialization}}
)";

    return TemplateLoadAndRender(source, value_map);
}

std::string MaskedCBlockTransferDesc::Emit()
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
    {{scalar_per_vector}}, // scalar_per_vector
    {% if causal_mask == 1 %}
    ck::tensor_operation::device::MaskingSpecialization::MaskOutUpperTriangle // causal_mask
    {% else %}
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled // causal_mask
    {% endif %}
)";

    return TemplateLoadAndRender(source, value_map);
}

GemmOperation::GemmOperation(GemmOperationKind                 operation_kind,
                             GemmKernelType                    kernel_type,
                             TensorDesc                        a_tensor_desc,
                             TensorDesc                        b_tensor_desc,
                             TensorDesc                        c_tensor_desc,
                             DataType                          accumulator_type,
                             TensorOperation                   a_element_op,
                             TensorOperation                   b_element_op,
                             TensorOperation                   epilogue_op,
                             GemmSpecialization                gemm_specialization,
                             GemmTileDesc                      tile_desc,
                             BlockTransferDesc                 a_block_transfer,
                             BlockTransferDesc                 b_block_transfer,
                             std::optional<CBlockTransferDesc> c_block_transfer,
                             std::optional<BlockTransferDesc>  b1_block_transfer,
                             std::vector<DataType>             ds_dtype,
                             std::vector<LayoutType>           ds_layout,
                             DataType                          e_dtype):
    operation_kind_(operation_kind),
    kernel_type_(kernel_type),
    a_tensor_desc_(a_tensor_desc),
    b_tensor_desc_(b_tensor_desc),
    c_tensor_desc_(c_tensor_desc),
    accumulator_type_(accumulator_type),
    a_element_op_(a_element_op),
    b_element_op_(b_element_op),
    epilogue_op_(epilogue_op),
    gemm_specialization_(gemm_specialization),
    tile_desc_(tile_desc),
    a_block_transfer_(a_block_transfer),
    b_block_transfer_(b_block_transfer),
    c_block_transfer_(c_block_transfer),
    b1_block_transfer_(b1_block_transfer),
    ds_dtype_(ds_dtype),
    ds_layout_(ds_layout),
    e_dtype_(e_dtype)
{
}

std::string GemmOperation::GetConfigName()
{
    std::string io_name = Sprintf("{}_{}{}{}{}_{}{}{}_{}",
                                  g_gemm_operation_kind_names.find(operation_kind_)->second,
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

    std::string tile_name = operation_kind_ == GemmOperationKind::BatchGemmSoftmaxGemmPermute ?
                                attn_tile_desc_.GetConfigName() + extra_tile :
                                tile_desc_.GetConfigName() + extra_tile;

    std::string casual_name =
        operation_kind_ == GemmOperationKind::BatchGemmSoftmaxGemmPermute ?
            g_short_tensor_operation_names_map.find(mask_c_block_transfer_->causal_mask_)->second :
            "";

    std::string config_name = Sprintf(
        "{}_{}_{}", io_name, tile_name, g_short_tensor_operation_names_map.find(c_element_op_)->second + casual_name);

    return config_name;
}

std::string GemmOperation::Emit()
{
    std::string source = R"( 
using {{name}} = {{kernel_type}}<
{% if kernel_type_value==0 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    {{CShuffleDType}},
{% elif kernel_type_value==1 %}
    {{ALayout}},
    {{BLayout}},
    ck::Tuple<{{DsLayout}}>, // DsLayout
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    {{CShuffleDType}},
    ck::Tuple<{{DsDType}}>, // DsType
    {{CDType}},
{% elif kernel_type_value==2 %}
    {{ADType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
{% elif kernel_type_value==3 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    {{CShuffleDType}},
    {{CDType}},
{% elif kernel_type_value==4 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    float, // CShuffleDType
    ck::half_t,
    ck::half_t,
{% elif kernel_type_value==5 %}
    {% if gemm_kind == "gemm_permute_m2n3" %}
    1, 2, 3, 1, // permute m2n3
    {% elif gemm_kind == "gemm_permute_m3n2" %}
    1, 3, 2, 1, // permute m3n2
    {% endif %}
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    {{CDType}},
    {% if "PassThrough" in C_elem_op %}
    ck::Tuple<>,
    {% else %}
    ck::Tuple<{{CDType}}>,
    {% endif %}
    {{CDType}},
{% elif kernel_type_value == 6 %}
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    float, // CShuffleDType,
{% elif kernel_type_value  == 7 %}
    2, 1, 1, 1, 1,
    {{ADType}},
    {{BDType}},
    {{BDType}},
    {{CDType}},
    ck::Tuple<>, // Acc0BiasDataType
    ck::Tuple<>, // Acc1BiasDataType
    {{AccDType}},
    float, // CShuffleDType,
{% elif kernel_type_value ==8 %}
    {{ALayout}},
    {{BLayout}},
    ck::Tuple<{{DsLayout}}>, // DsLayout
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{AccDType}},
    {{CShuffleDType}},
    ck::Tuple<{{DsDType}}>, // DsType
    {{EDType}},
{% elif kernel_type_value ==9 %}
    {{ADType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
{% endif %}
{% if kernel_type_value in [6, 7] %}
    {{A_elem_op}},
    {{B_elem_op}},
    ck::tensor_operation::element_wise::Scale,
{% else %}
    {{A_elem_op}},
{% endif %}
    {{B_elem_op}},
    {{C_elem_op}},
{% if kernel_type_value!=2 %}
    {{GemmSpecialization}},
    {% if kernel_type_value==5 %}
    ck::tensor_operation::device::TensorSpecialization::Packed,
    ck::tensor_operation::device::TensorSpecialization::Packed,
    ck::tensor_operation::device::TensorSpecialization::Default,
    {% elif kernel_type_value == 7 %}
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    {% endif %}
    {% if kernel_type_value!=9 %}
    1,
    {% endif %}
{% endif %}
    {{tile_config}}
    {{a_block_transfer}}
    {{b_block_transfer}}
{% if kernel_type_value in [6, 7] %}
    {{b1_block_transfer}}
{% endif %}
{% if kernel_type_value!=2 %}
    {{c_block_transfer}}
{% else %}
    7, // src_dst_vector_dim
    1 // dst_scalar_per_vector
{% endif %}
    >;
)";

    std::string ds_dtype_value = "";
    if (!ds_dtype_.empty()) {
        for (const auto& d_dtype : ds_dtype_) {
            ds_dtype_value += DataTypeToString(d_dtype) + ",";
        }
        ds_dtype_value.pop_back();
    }

    std::string ds_layout_value = "";
    if (!ds_layout_.empty()) {
        for (auto d_layout : ds_layout_) {
            ds_layout_value += g_layout_tag.find(d_layout)->second + ",";
        }
        ds_layout_value.pop_back();
    }

    std::string e_dtype_value = e_dtype_ == DataType::UNDEFINED ? "" : DataTypeToString(e_dtype_);

    auto name_value = this->GetConfigName();
    auto tile_config_value =
        operation_kind_ == GemmOperationKind::BatchGemmSoftmaxGemmPermute ? attn_tile_desc_.Emit() : tile_desc_.Emit();
    auto a_block_transfer_value = a_block_transfer_.Emit();
    auto b_block_transfer_value = b_block_transfer_.Emit();

    auto b1_block_transfer_value = b1_block_transfer_.has_value() ? b1_block_transfer_->Emit() : "";

    auto c_block_transfer_value =
        c_block_transfer_.has_value() ? c_block_transfer_->Emit() : mask_c_block_transfer_->Emit();

    jinja2::ValuesMap value_map{{"name", name_value},
                                {"gemm_kind", g_gemm_operation_kind_names.find(operation_kind_)->second},
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

}  // namespace lightinfer
