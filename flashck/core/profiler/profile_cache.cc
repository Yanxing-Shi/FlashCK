#include "flashck/core/profiler/profile_cache.h"

#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/jinjia2_utils.h"
#include "flashck/core/utils/log.h"
#include "flashck/core/utils/printf.h"

namespace flashck {

ProfileCacheDB::ProfileCacheDB(const std::string&           device_name,
                               const std::filesystem::path& path,
                               const std::string&           port):
    device_name_(device_name), path_(path)
{
    mode_ = CacheMode::kLocal;

    CHECK_SQLITE(
        sqlite3_open_v2(path.string().c_str(), &cache_db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr),
        cache_db_);

    CreateGemmTable();
    CreateNormTable();
    CreateFmhaTable();
    CreateEmbeddingTable();
}

void ProfileCacheDB::CreateGemmTable()
{
    auto sql = TemplateLoadAndRender(g_gemm_init_source, {{"dev", device_name_}});
    CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

void ProfileCacheDB::CreateNormTable()
{
    auto sql = TemplateLoadAndRender(g_layer_norm_init_source, {{"dev", device_name_}});
    CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

void ProfileCacheDB::CreateFmhaTable()
{
    auto sql = TemplateLoadAndRender(g_fmha_init_source, {{"dev", device_name_}});
    CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

void ProfileCacheDB::CreateEmbeddingTable()
{
    auto sql = TemplateLoadAndRender(g_embedding_init_source, {{"dev", device_name_}});
    CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

bool ProfileCacheDB::TableExists(const GenOperationKind& op_kind)
{
    const auto it = g_gen_operation_kind_names.find(op_kind);
    if (it == g_gen_operation_kind_names.end()) {
        LOG(ERROR) << "Invalid operation kind: " << static_cast<int>(op_kind);
        return false;
    }

    std::string table_name = device_name_ + "_" + it->second;
    auto        sql        = TemplateLoadAndRender(g_check_table_exist_source, {{"table_name", table_name}});

    CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, sql, -1, &stmt, nullptr));

    const bool is_exists = (sqlite3_step(stmt) == SQLITE_ROW);

    LOG(INFO) << "Table [" << table_name << "] " << (is_exists ? "exists" : "does not exist");
}

std::tuple<std::string, int64_t> ProfileCacheDB::Query(const std::string& sql)
{
    std::string query_algo = "null";
    int64_t     split_k    = -1;

    if (mode_ == CacheMode::kLocal) {
        sqlite3_stmt* stmt;
        CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, sql.c_str(), -1, &stmt, NULL), cache_db_);

        while (true) {
            auto step_result = sqlite3_step(stmt);

            VLOG(1) << "step_result: " << step_result;

            if (step_result == SQLITE_DONE) {
                break;
            }

            if (step_ret != SQLITE_ROW) {
                LOG(ERROR) << "SQL error: " << sqlite3_errstr(step_ret);
                return {"null", -1};
            }

            query_algo = sqlite3_column_type(stmt.stmt, 0) == SQLITE_TEXT ?
                             reinterpret_cast<const char*>(sqlite3_column_text(stmt.stmt, 0)) :
                             "null";
            split_k    = sqlite3_column_type(stmt.stmt, 1) != SQLITE_NULL ? sqlite3_column_int64(stmt.stmt, 1) : -1;
        };
    }

    VLOG(1) << "query_algo: " << query_algo;

    return {query_algo.empty() ? "null" : query_algo, split_k};
}

std::tuple<std::string, int64_t> ProfileCacheDB::QueryGemm(const GemmQueryEntry& query)
{

    jinja2::ValuesMap gemm_query_value_map{{"device_name", device_name_},
                                           {"a_dtype", query.a_dtype_},
                                           {"b_dtype", query.b_dtype_},
                                           {"c_dtype", query.c_dtype_},
                                           {"acc_dtype", query.acc_dtype_},
                                           {"layout", query.layout_},
                                           {"op_kind", query.op_kind_},
                                           {"epilogue", query.epilogue_},
                                           {"pshape", query.pshape_},
                                           {"exec_entry_sha1", query.exec_entry_sha1_}};

    auto sql = TemplateLoadAndRender(g_gemm_query_source, gemm_query_value_map);
    VLOG(1) << sql;
    return Query(sql);
}

// a function to query normalization op epilogue from cache
std::tuple<std::string, int64_t> ProfileCacheDB::QueryNorm(const NormQueryEntry& query)
{

    jinja2::ValuesMap norm_query_value_map{{"device_name", device_name_},
                                           {"x_dtype", query.x_dtype_},
                                           {"y_dtype", query.y_dtype_},
                                           {"smooth_scale_dtype", query.smooth_scale_dtype_},
                                           {"y_scale_dtype", query.y_scale_dtype_},
                                           {"op_kind", query.op_kind_},
                                           {"epilogue", query.epilogue_},
                                           {"exec_entry_sha1", query.exec_entry_sha1_},
                                           {"fused_add", query.fused_add_},
                                           {"fused_quant", query.fused_quant_}};

    auto sql = TemplateLoadAndRender(g_norm_query_source, norm_query_value_map);
    VLOG(1) << sql;
    return Query(sql);
}

std::tuple<std::string, int64_t> ProfileCacheDB::QueryFmha(const FmhaQueryEntry& query)
{

    jinja2::ValuesMap fmha_query_value_map{{"dev", device_name_},
                                           {"dtype", query.dtype_},
                                           {"mask", query.mask_},
                                           {"bias", query.bias_},
                                           {"mode", query.mode_},
                                           {"rotary_dim", query.rotary_dim_},
                                           {"paged_block_size", query.paged_block_size_},
                                           {"use_batch_cache_idx", static_cast<int>(query.use_batch_cache_idx_)},
                                           {"op_name", query.op_name_},
                                           {"device", device_name_},
                                           {"epilogue", query.epilogue_},
                                           {"exec_entry_sha1", query.exec_entry_sha1_}};

    auto sql = TemplateLoadAndRender(g_fmha_query_source, fmha_query_value_map);
    VLOG(1) << sql;
    return Query(sql);
}

std::tuple<std::string, int64_t> ProfileCacheDB::QueryEmbedding(const EmbeddingQueryEntry& query)
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
    VLOG(1) << sql;
    return Query(sql);
}

int64_t ProfileCacheDB::CheckIfInsert(const std::string& query_sql)
{
    int64_t n_query_result = 0;

    if (mode_ == CacheMode::kLocal) {
        sqlite3_stmt* stmt;
        CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, query_sql.c_str(), -1, &stmt, nullptr), cache_db_);
        while (true) {
            auto step_result = sqlite3_step(stmt);

            if (step_result == SQLITE_DONE) {
                break;
            }

            n_query_result += 1;
        };
        CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    return n_query_result;
}

void ProfileCacheDB::InsertGemm(const GemmRecordEntry& record)
{

    jinja2::ValuesMap gemm_query_value_map{{"dev", device_name_},
                                           {"a_dtype", record.a_dtype_},
                                           {"b_dtype", record.b_dtype_},
                                           {"c_dtype", record.c_dtype_},
                                           {"acc_dtype", record.acc_dtype_},
                                           {"layout", record.layout_},
                                           {"op_name", record.op_name_},
                                           {"device", device_name_},
                                           {"epilogue", record.epilogue_},
                                           {"pshape", record.pshape_},
                                           {"exec_entry_sha1", record.exec_entry_sha1_}};

    jinja2::ValuesMap gemm_insert_value_map{{"dev", device_name_}};

    auto query_sql  = TemplateLoadAndRender(g_gemm_query_source, gemm_query_value_map);
    auto insert_sql = TemplateLoadAndRender(g_gemm_insert_source, gemm_insert_value_map);

    const int n_query_result = CheckIfInsert(query_sql);

    VLOG(1) << "query_sql: " << query_sql;
    VLOG(1) << "insert_sql: " << insert_sql;
    VLOG(1) << "n_query_result: " << n_query_result;

    if (!n_query_result) {
        sqlite3_stmt* stmt;
        CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

        int column_count = 1;

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.a_dtype_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.b_dtype_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.c_dtype_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.acc_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.layout_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.split_k_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.pshape_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            VLOG(1) << "Insert gemm success";
        }
        else {
            VLOG(1) << "Insert failed";
        }

        CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    else {
        VLOG(1) << "Ignore repeat profile_record:" << query_sql;
    }
}

// a function to insert norm op into cache
void ProfileCacheDB::InsertNorm(const NormRecordEntry& record)
{
    jinja2::ValuesMap layer_norm_query_value_map{{"dev", device_name_},
                                                 {"x_dtype", record.x_dtype_},
                                                 {"y_dtype", record.y_dtype_},
                                                 {"smooth_scale_dtype", record.smooth_scale_dtype_},
                                                 {"y_scale_dtype", record.y_scale_dtype_},
                                                 {"op_name", record.op_name_},
                                                 {"device", device_name_},
                                                 {"epilogue", record.epilogue_},
                                                 {"exec_entry_sha1", record.exec_entry_sha1_},
                                                 {"fused_add", record.fused_add_},
                                                 {"fused_quant", record.fused_quant_}};

    jinja2::ValuesMap layer_norm_insert_value_map{{"dev", device_name_}};

    auto query_sql  = TemplateLoadAndRender(g_norm_query_source, layer_norm_query_value_map);
    auto insert_sql = TemplateLoadAndRender(g_norm_insert_source, layer_norm_insert_value_map);

    const int64_t n_query_result = CheckIfInsert(query_sql);

    VLOG(1) << "query_sql: " << query_sql;
    VLOG(1) << "insert_sql: " << insert_sql;
    VLOG(1) << "n_query_result: " << n_query_result;

    if (!n_query_result) {
        sqlite3_stmt* stmt;
        CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

        int64_t column_count = 1;

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.x_dtype_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.y_dtype_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.smooth_scale_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.y_scale_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.fused_add_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.fused_quant_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            VLOG(1) << "Insert tile layer norm success";
        }
        else {
            VLOG(1) << "Insert failed";
        }

        CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    else {
        LOG(WARNING) << "Ignore repeat profile_record:" << query_sql;
    }
}

// a function to insert norm op into cache
void ProfileCacheDB::InsertFmha(const FmhaRecordEntry& record)
{
    jinja2::ValuesMap fmha_query_value_map{{"dev", device_name_},
                                           {"dtype", record.dtype_},
                                           {"mask", record.mask_},
                                           {"bias", record.bias_},
                                           {"mode", record.mode_},
                                           {"rotary_dim", record.rotary_dim_},
                                           {"paged_block_size", record.paged_block_size_},
                                           {"use_batch_cache_idx", static_cast<int>(record.use_batch_cache_idx_)},
                                           {"op_name", record.op_name_},
                                           {"device", device_name_},
                                           {"epilogue", record.epilogue_},
                                           {"exec_entry_sha1", record.exec_entry_sha1_}};

    jinja2::ValuesMap fmha_insert_value_map{{"dev", device_name_}};

    auto query_sql  = TemplateLoadAndRender(g_fmha_query_source, fmha_query_value_map);
    auto insert_sql = TemplateLoadAndRender(g_fmha_insert_source, fmha_insert_value_map);

    const int64_t n_query_result = CheckIfInsert(query_sql);

    VLOG(1) << "query_sql: " << query_sql;
    VLOG(1) << "insert_sql: " << insert_sql;
    VLOG(1) << "n_query_result: " << n_query_result;

    if (!n_query_result) {
        sqlite3_stmt* stmt;
        CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

        int64_t column_count = 1;

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.dtype_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.mask_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.bias_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.mode_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.rotary_dim_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.paged_block_size_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, static_cast<int>(record.use_batch_cache_idx_)), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.num_splits_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            VLOG(1) << "Insert fmha success";
        }
        else {
            VLOG(1) << "Insert failed";
        }

        CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    else {
        LOG(WARNING) << "Ignore repeat profile_record:" << query_sql;
    }
}

void ProfileCacheDB::InsertEmbedding(const EmbeddingRecordEntry& record)
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

    auto query_sql  = TemplateLoadAndRender(g_embedding_query_source, embedding_query_value_map);
    auto insert_sql = TemplateLoadAndRender(g_embedding_insert_source, {{"device_name", device_name_}});

    const int64_t n_query_result = CheckIfInsert(query_sql);

    VLOG(1) << "query_sql: " << query_sql;
    VLOG(1) << "insert_sql: " << insert_sql;
    VLOG(1) << "n_query_result: " << n_query_result;

    if (!n_query_result) {
        sqlite3_stmt* stmt;
        CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr),
                     cache_db_,
                     "prepare insert sql for embedding");

        int64_t column_count = 1;

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_,
                     "bind text for embedding");

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.num_embeddings_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.vocab_size_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.type_vocab_size_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.max_position_embeddings_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.embedding_dims_), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.emb_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.index_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.gamma_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.beta_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.acc_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.y_dtype_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_op_.c_str(), -1, SQLITE_TRANSIENT),
                     cache_db_);

        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);
        CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            LOG(INFO) << "Insert embedding success";
        }
        else {
            LOG(ERROR) << "Insert failed";
        }

        CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    else {
        LOG(WARNING) << "Ignore repeat profile_record:" << query_sql;
    }
}

ProfileCacheDB::~ProfileCacheDB()
{
    CHECK_SQLITE(sqlite3_close_v2(cache_db_), cache_db_);
}

}  // namespace flashck