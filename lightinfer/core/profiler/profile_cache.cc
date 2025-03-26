#include "lightinfer/core/profiler/profile_cache.h"

#include "lightinfer/core/utils/file_utils.h"
#include "lightinfer/core/utils/jinjia2_utils.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/printf.h"

namespace lightinfer {

ProfileCacheDB::ProfileCacheDB(const std::string&           device_name,
                               const std::filesystem::path& path,
                               const std::string&           uri,
                               const std::string&           port):
    device_name_(device_name), path_(path), uri_(uri), port_(port)
{
    if (!uri.empty()) {
        mode_ = CacheModeType::Remote;
    }
    else {
        mode_ = CacheModeType::Local;
        LI_CHECK_SQLITE(
            sqlite3_open_v2(path.string().c_str(), &cache_db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr),
            cache_db_);

        CreateGemmTable();
        CreateNormTable();
        CreateFmhaTable();
        CreateEmbeddingTable();
    }
}

// Creates gemm table
void ProfileCacheDB::CreateGemmTable()
{
    auto sql = TemplateLoadAndRender(g_gemm_init_source, {{"dev", device_name_}});
    LI_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

// Create Norm Table
void ProfileCacheDB::CreateNormTable()
{
    auto sql = TemplateLoadAndRender(g_layer_norm_init_source, {{"dev", device_name_}});
    LI_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

void ProfileCacheDB::CreateFmhaTable()
{
    auto sql = TemplateLoadAndRender(g_fmha_init_source, {{"dev", device_name_}});
    LI_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

void ProfileCacheDB::CreateEmbeddingTable()
{
    auto sql = TemplateLoadAndRender(g_embedding_init_source, {{"dev", device_name_}});
    LI_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
}

bool ProfileCacheDB::TableExists(const GenOperationKind& op_kind)
{

    std::string table_name = Sprintf("{}_{}", device_name_, g_gen_operation_kind_names.find(op_kind)->second);

    jinja2::ValuesMap check_table_value_map{{"table_name", table_name}};
    auto              sql = TemplateLoadAndRender(g_check_table_exist_source, check_table_value_map);

    bool is_table_exist;

    int (*callback)(void*, int, char**, char**) =
        LambdaToPointer([&](void* para, int column_count, char** column_value, char** column_name) -> int {
            is_table_exist = column_count >= 1 ? true : false;
            return 0;
        });

    LI_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), callback, NULL, NULL), cache_db_);

    if (is_table_exist) {
        LOG(INFO) << "Table: " << table_name << " exists in the db";
        return true;
    }
    else {
        LOG(INFO) << "Table: " << table_name << " does not exist in the db";
        return false;
    }
}

std::tuple<std::string, int64_t> ProfileCacheDB::Query(const std::string& sql)
{
    std::string query_algo;
    int64_t     split_k;

    if (mode_ == CacheModeType::Local) {
        sqlite3_stmt* stmt;
        LI_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, sql.c_str(), -1, &stmt, NULL), cache_db_);

        int64_t column_count = 0;

        while (true) {
            auto step_result = sqlite3_step(stmt);

            VLOG(1) << "step_result: " << step_result;

            if (step_result == SQLITE_DONE) {
                break;
            }

            query_algo = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, column_count++)));

            if (sqlite3_column_type(stmt, column_count) != SQLITE_NULL) {
                split_k = sqlite3_column_int(stmt, column_count++);
            }
            else {
                split_k = -1;
            }
        };
        LI_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    VLOG(1) << "query_algo: " << query_algo;

    if (query_algo.empty()) {
        return std::make_tuple("null", -1);
    }

    return std::make_tuple(query_algo, split_k);
}

std::tuple<std::string, int64_t> ProfileCacheDB::QueryGemm(const GemmQueryEntry& query)
{

    jinja2::ValuesMap gemm_query_value_map{{"dev", device_name_},
                                           {"a_dtype", query.a_dtype_},
                                           {"b_dtype", query.b_dtype_},
                                           {"c_dtype", query.c_dtype_},
                                           {"acc_dtype", query.acc_dtype_},
                                           {"layout", query.layout_},
                                           {"op_name", query.op_name_},
                                           {"device", device_name_},
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

    jinja2::ValuesMap tile_layer_norm_query_value_map{{"dev", device_name_},
                                                      {"x_dtype", query.x_dtype_},
                                                      {"y_dtype", query.y_dtype_},
                                                      {"smooth_scale_dtype", query.smooth_scale_dtype_},
                                                      {"y_scale_dtype", query.y_scale_dtype_},
                                                      {"op_name", query.op_name_},
                                                      {"device", device_name_},
                                                      {"epilogue", query.epilogue_},
                                                      {"exec_entry_sha1", query.exec_entry_sha1_},
                                                      {"fused_add", query.fused_add_},
                                                      {"fused_quant", query.fused_quant_}};

    auto sql = TemplateLoadAndRender(g_norm_query_source, tile_layer_norm_query_value_map);
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

    if (mode_ == CacheModeType::Local) {
        sqlite3_stmt* stmt;
        LI_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, query_sql.c_str(), -1, &stmt, nullptr), cache_db_);
        while (true) {
            auto step_result = sqlite3_step(stmt);

            if (step_result == SQLITE_DONE) {
                break;
            }

            n_query_result += 1;
        };
        LI_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
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
        LI_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

        int column_count = 1;

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.a_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.b_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.c_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.acc_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.layout_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.split_k_), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.pshape_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            VLOG(1) << "Insert gemm success";
        }
        else {
            VLOG(1) << "Insert failed";
        }

        LI_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
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
        LI_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

        int64_t column_count = 1;

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.x_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.y_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(
            sqlite3_bind_text(stmt, column_count++, record.smooth_scale_dtype_.c_str(), -1, SQLITE_TRANSIENT),
            cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.y_scale_dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.fused_add_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.fused_quant_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            VLOG(1) << "Insert tile layer norm success";
        }
        else {
            VLOG(1) << "Insert failed";
        }

        LI_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
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
        LI_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

        int64_t column_count = 1;

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.dtype_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.mask_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.bias_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.mode_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.rotary_dim_), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.paged_block_size_), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, static_cast<int>(record.use_batch_cache_idx_)),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.op_name_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.epilogue_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
                        cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, record.num_splits_), cache_db_);

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            VLOG(1) << "Insert fmha success";
        }
        else {
            VLOG(1) << "Insert failed";
        }

        LI_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
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

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.exec_entry_sha1_.c_str(), -1, SQLITE_TRANSIENT),
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

        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);
        LI_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, record.algo_.c_str(), -1, SQLITE_TRANSIENT), cache_db_);

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

ProfileCacheDB::~ProfileCacheDB()
{
    LI_CHECK_SQLITE(sqlite3_close_v2(cache_db_), cache_db_);
}

}  // namespace lightinfer