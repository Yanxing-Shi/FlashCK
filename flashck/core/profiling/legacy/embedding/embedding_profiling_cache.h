#pragma once

namespace flashck {

constexpr std::string kEmbeddingInitSource = R"sql(
CREATE TABLE IF NOT EXISTS {{device_name}}_embedding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exec_entry VARCHAR(8192),
    num_embeddings INTEGER,
    vocab_size INTEGER,
    type_vocab_size INTEGER,
    max_position_embeddings INTEGER,
    embedding_dims INTEGER,
    emb_dtype VARCHAR(8),
    index_dtype VARCHAR(8),
    gamma_dtype VARCHAR(8),
    beta_dtype VARCHAR(8),
    acc_dtype VARCHAR(8),
    out_dtype VARCHAR(8),
    op_name VARCHAR(512),
    algo VARCHAR(512),
    duration FLOAT DEFAULT -1
    );
)sql";

constexpr std::string kEmbeddingQuerySource = R"sql(
SELECT algo FROM {{device_name}}_embedding WHERE
    exec_entry='{{exec_entry}}' AND
    num_embeddings={{num_embeddings}} AND
    vocab_size={{vocab_size}} AND
    type_vocab_size={{type_vocab_size}} AND
    max_position_embeddings={{max_position_embeddings}} AND
    embedding_dims={{embedding_dims}} AND
    emb_dtype='{{emb_dtype}}' AND
    index_dtype='{{index_dtype}}' AND
    acc_dtype='{{acc_dtype}}' AND
    gamma_dtype='{{gamma_dtype}}' AND
    beta_dtype='{{beta_dtype}}' AND
    out_dtype='{{out_dtype}}' AND
    op_name='{{op_name}}';
)sql";

class EmbeddingProfilingCache: public ProfilingCacheBase<EmbeddingProfilingCache> {

    void CreateTable()
    {
        auto sql = TemplateLoadAndRender(g_embedding_init_source, {{"dev", device_name_}});
        FC_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
    }

    template<typename QueryType>
    std::tuple<std::string, int64_t> Query(const QueryType& query)
    {
        jinja2::ValuesMap embedding_query_value_map{{"dev", device_name_},
                                                    {"num_embeddings", query.num_embeddings_},
                                                    {"vocab_size", query.vocab_size_},
                                                    {"type_vocab_size", query.type_vocab_size_},
                                                    {"max_position_embeddings", query.max_position_embeddings_},
                                                    {"embedding_dims", query.embedding_dims_},
                                                    {"emb_dtype", query.emb_dtype_},
                                                    {"index_dtype", query.index_dtype_},
                                                    {"gamma_dtype", query.gamma_dtype_},
                                                    {"beta_dtype", query.beta_dtype_},
                                                    {"acc_dtype", query.acc_dtype_},
                                                    {"out_dtype", query.y_dtype_},
                                                    {"op_name", query.op_name_},
                                                    {"epilogue", query.epilogue_op_},
                                                    {"device", device_name_},
                                                    {"exec_entry_sha1", query.exec_entry_sha1_}};

        auto sql = TemplateLoadAndRender(g_embedding_query_source, embedding_query_value_map);
        VLOG(1) << "EmbeddingProfilingCache query sql command: " << sql;
        return QueryCache(sql);
    }

    template<typename RecordType>
    void InsertEmbedding(const RecordType& record)
    {
        jinja2::ValuesMap embedding_query_value_map{{"dev", device_name_},
                                                    {"num_embeddings", record.num_embeddings_},
                                                    {"vocab_size", record.vocab_size_},
                                                    {"type_vocab_size", record.type_vocab_size_},
                                                    {"max_position_embeddings", record.max_position_embeddings_},
                                                    {"embedding_dims", record.embedding_dims_},
                                                    {"emb_dtype", record.emb_dtype_},
                                                    {"index_dtype", record.index_dtype_},
                                                    {"gamma_dtype", record.gamma_dtype_},
                                                    {"beta_dtype", record.beta_dtype_},
                                                    {"acc_dtype", record.acc_dtype_},
                                                    {"out_dtype", record.y_dtype_},
                                                    {"op_name", record.op_name_},
                                                    {"epilogue", record.epilogue_op_},
                                                    {"device", device_name_},
                                                    {"exec_entry_sha1", record.exec_entry_sha1_}};

        jinja2::ValuesMap embedding_insert_value_map{{"dev", device_name_}};

        auto query_sql  = TemplateLoadAndRender(g_embedding_query_source, embedding_query_value_map);
        auto insert_sql = TemplateLoadAndRender(g_embedding_insert_source, embedding_insert_value_map);

        const int64_t n_query_result = CheckIfInsert(query_sql);

        VLOG(1) << "query_sql: " << query_sql;
        VLOG(1) << "insert_sql: " << insert_sql;
        VLOG(1) << "n_query_result: " << n_query_result;

        if (!n_query_result) {
            sqlite3_stmt* stmt;
            LI_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

            int64_t column_count = 1;

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(
                sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.num_embeddings_), cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.vocab_size_), cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.type_vocab_size_), cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.max_position_embeddings_), cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.embedding_dims_), cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.emb_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.index_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.gamma_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.beta_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.acc_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.y_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_op_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);
            LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT),
                            cache_db_);

            if (sqlite3_step(stmt) == SQLITE_DONE) {
                LOG(INFO) << "Insert embedding success";
            }
            else {
                LOG(ERROR) << "Insert failed";
            }

            LI_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
        }

        else {
            LOG(WARNING) << "Ignore repeat profile_record:" << query_sql;
        }
    }
};

}  // namespace flashck