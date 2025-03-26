#include "lightinfer/core/profiler/embedding_operation.h"

#include "lightinfer/core/utils/jinjia2_utils.h"
#include "lightinfer/core/utils/printf.h"

namespace lightinfer {

EmbeddingTileDesc::EmbeddingTileDesc(int64_t block_size,
                                     int64_t dim_cluster_size,
                                     int64_t row_cluster_size,
                                     int64_t dim_per_block,
                                     int64_t row_per_block,
                                     int64_t dim_thread_size,
                                     int64_t row_vector_size):
    block_size_(block_size),
    dim_cluster_size_(dim_cluster_size),
    row_cluster_size_(row_cluster_size),
    dim_per_block_(dim_per_block),
    row_per_block_(row_per_block),
    dim_thread_size_(dim_thread_size),
    row_vector_size_(row_vector_size)
{
}

std::string EmbeddingTileDesc::GetConfigName()
{
    return Sprintf("{}_{}_{}_{}_{}_{}_{}",
                   block_size_,
                   dim_cluster_size_,
                   row_cluster_size_,
                   dim_per_block_,
                   row_per_block_,
                   dim_thread_size_,
                   row_vector_size_);
}

std::string EmbeddingTileDesc::Emit()
{
    std::string       source = R"(
{% for value in param %}
{% if value!=0 %}
    {{value}},
{% endif %}
{% endfor %}
)";
    jinja2::ValuesMap value_map{{"param",
                                 jinja2::ValuesList{block_size_,
                                                    dim_cluster_size_,
                                                    row_cluster_size_,
                                                    dim_per_block_,
                                                    row_per_block_,
                                                    dim_thread_size_,
                                                    row_vector_size_}}};

    return TemplateLoadAndRender(source, value_map);
}

std::string EmbeddingOperation::GetConfigName()
{
    return num_elements_ == 3 ? Sprintf("{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                                        g_embedding_operation_kind_names.find(operation_kind_)->second,
                                        g_short_tensor_operation_names_map.find(epilogue_op_)->second,
                                        vocab_size_,
                                        type_vocab_size_,
                                        max_position_embeddings_,
                                        tile_desc_.GetConfigName(),
                                        embedding_dims_,
                                        DataTypeToShortString(emb_dtype_),
                                        DataTypeToShortString(index_dtype_),
                                        DataTypeToShortString(gamma_dtype_),
                                        DataTypeToShortString(beta_dtype_),
                                        DataTypeToShortString(acc_dtype_),
                                        DataTypeToShortString(y_dtype_)) :
                                Sprintf("{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                                        g_embedding_operation_kind_names.find(operation_kind_)->second,
                                        g_short_tensor_operation_names_map.find(epilogue_op_)->second,
                                        vocab_size_,
                                        tile_desc_.GetConfigName(),
                                        embedding_dims_,
                                        DataTypeToShortString(emb_dtype_),
                                        DataTypeToShortString(index_dtype_),
                                        DataTypeToShortString(gamma_dtype_),
                                        DataTypeToShortString(beta_dtype_),
                                        DataTypeToShortString(acc_dtype_),
                                        DataTypeToShortString(y_dtype_));
}

std::string EmbeddingOperation::Emit()
{
    std::string source = R"(
using {{name}} = {{embedding_kernel_type}}<
    {{EmbType}}, // embedding type
    {{IndexType}}, // index type
    {{GammaDataType}}, // gamma type
    {{BetaDataType}}, // beta type
    {{AccDataType}}, // acc type
    {{OutType}}, // output type
    EmbElementwiseOperation,
    {{tile_config}}
    {{num_elements}}
    >;
    )";

    jinja2::ValuesMap value_map{{"name", GetConfigName()},
                                {"EmbType", DataTypeToString(emb_dtype_)},
                                {"IndexType", DataTypeToString(index_dtype_)},
                                {"GammaDataType", DataTypeToString(gamma_dtype_)},
                                {"BetaDataType", DataTypeToString(beta_dtype_)},
                                {"AccDataType", DataTypeToString(acc_dtype_)},
                                {"OutType", DataTypeToString(y_dtype_)},
                                {"embedding_kernel_type", g_embedding_kernel_tag.find(embedding_kernel_type_)->second},
                                {"tile_config", tile_desc_.Emit()},
                                {"num_elements", num_elements_}};

    return TemplateLoadAndRender(source, value_map);
}

}  // namespace lightinfer