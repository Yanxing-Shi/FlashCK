#include "flashck/core/profiling/legacy/embedding/embedding_codegen.h"

namespace flashck {

std::string EmbeddingTileDesc::GetConfigName() const
{
    return Sprintf(
        "{block_size}_{dim_cluster_size}_{row_cluster_size}_{dim_per_block}_{row_per_block}_{dim_thread_size}_{row_vector_size}",
        fmt::arg(block_size, block_size_),
        fmt::arg(dim_cluster_size, dim_cluster_size_),
        fmt::arg(row_cluster_size, row_cluster_size_),
        fmt::arg(dim_per_block, dim_per_block_),
        fmt::arg(row_per_block, row_per_block_),
        fmt::arg(dim_thread_size, dim_thread_size_),
        fmt::arg(row_vector_size, row_vector_size_));
}

std::string EmbeddingTileDesc::Emit() const
{
    constexpr std::string source = R"(
{% for value in param %}
{% if value!=0 %}
    {{value}},
{% endif %}
{% endfor %}
)";
    jinja2::ValuesMap     value_map{{"param",
                                     jinja2::ValuesList{block_size_,
                                                    dim_cluster_size_,
                                                    row_cluster_size_,
                                                    dim_per_block_,
                                                    row_per_block_,
                                                    dim_thread_size_,
                                                    row_vector_size_}}};

    return TemplateLoadAndRender(source, value_map);
}

std::string EmbeddingCodegen::GetConfigName() const
{

    return Sprintf(
        "{op_kind}_{vocab_size}_{type_vocab_size}_{max_position_embeddings}_{tile_desc}_{embedding_dims}_{emb_dtype}_{index_dtype}_{gamma_dtype}_{beta_dtype}_{acc_dtype}_{y_dtype}",
        fmt::arg(op_kind, kind_),
        fmt::arg(epilogue_op, epilogue_op_),
        fmt::arg(vocab_size, vocab_size_),
        fmt::arg(type_vocab_size, type_vocab_size_),
        fmt::arg(max_position_embeddings, max_position_embeddings_),
        fmt::arg(tile_desc, tile_desc_.GetConfigName()),
        fmt::arg(embedding_dims, embedding_dims_),
        fmt::arg(emb_dtype, DataTypeToShortString(emb_dtype_)),
        fmt::arg(index_dtype, DataTypeToShortString(index_dtype_)),
        fmt::arg(gamma_dtype, DataTypeToShortString(gamma_dtype_)),
        fmt::arg(beta_dtype, DataTypeToShortString(beta_dtype_)),
        fmt::arg(acc_dtype, DataTypeToShortString(acc_dtype_)),
        fmt::arg(y_dtype, DataTypeToShortString(y_dtype_)));
}

std::string EmbeddingCodegen::Emit()
{
    std::string source = R"(
using {{name}} = <>
    {{embedding_dtype}}, // embedding dtype
    {{index_dtype}}, // index dtype
    {{gamma_dtype}}, // gamma dtype
    {{beta_dtype}}, // beta dtype
    {{acc_dtype}}, // acc dtype
    {{y_dtype}}, // output dtype
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

}  // namespace flashck