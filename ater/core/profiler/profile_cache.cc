#include "ater/core/profiler/profile_cache.h"

#include "ater/core/utils/file_utils.h"
#include "ater/core/utils/jinjia2_utils.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/printf.h"

namespace ater {

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
        ATER_CHECK_SQLITE(
            sqlite3_open_v2(path.string().c_str(), &cache_db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr),
            cache_db_);
        gemm_cache_version_ = 1;  // suppose
        CreateGemmTable();
        // norm_cache_version_ = 1;  // suppose
        // CreateNormTable();
    }
}

// Gemm_cache_version
int ProfileCacheDB::GetGemmCacheVersion()
{
    return gemm_cache_version_;
}

// norm_cache_version
int ProfileCacheDB::GetNormCacheVersion()
{
    return norm_cache_version_;
}

// Creates gemm table
void ProfileCacheDB::CreateGemmTable()
{
    const int gemm_curr_cache_version = GetGemmCacheVersion();
    if (!TableExists("gemm", gemm_curr_cache_version)) {
        LOG(INFO) << "Temporarily keeping the old gemm cache versions if exist";
        // FIXME : will delete unmatched version once we get into production
        // DeleteExistTable("gemm");

        LOG(INFO) << "Creating a new gemm table with " << gemm_curr_cache_version;

        jinja2::ValuesMap gemm_init_value_map{{"dev", device_name_}, {"version", gemm_curr_cache_version}};
        auto              sql = TemplateLoadAndRender(g_gemm_init_source, gemm_init_value_map);
        ATER_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
    }
}

// // Create Norm Table
// void CreateNormTable()
// {
//     int norm_curr_cache_version = GetNormCacheVersion();
//     if (TableExists("gemm", norm_curr_cache_version)) {
//         printf("Creating a new gemm table with %d", gemm_curr_cache_version);
//         jinja2::Template norm_tpl;
//         CheckTemplate(tpl.Load(g_norm_init_template));
//         jinja2::ValuesMap norm_init_value_map{{{"dev", device_name_}, {"version", gemm_curr_cache_version}}};
//         auto              sql = norm_tpl.RenderAsString(norm_init_value_map).value();
//         CheckSqlite(sqlite3_exec(cache_db_, sql.c_str(), nullptr, nullptr, nullptr), cache_db_);
//     }
// }

bool ProfileCacheDB::TableExists(const std::string& table_kind, const int table_version)
{
    std::string table_name = Sprintf("{}_{}_{}", device_name_, table_kind, table_version);

    jinja2::ValuesMap check_table_value_map{{"table_name", table_name}};
    auto              sql = TemplateLoadAndRender(g_check_table_exist_source, check_table_value_map);

    bool is_table_exist;

    int (*callback)(void*, int, char**, char**) =
        LambdaToPointer([&](void* para, int column_count, char** column_value, char** column_name) -> int {
            is_table_exist = column_count >= 1 ? true : false;
            return 0;
        });

    ATER_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), callback, NULL, NULL), cache_db_);

    if (is_table_exist) {
        LOG(INFO) << "Table: " << table_name << " exists in the db";
        return true;
    }
    else {
        LOG(INFO) << "Table: " << table_name << " does not exist in the db";
        return false;
    }
}

// Get gemm cache version
int ProfileCacheDB::GetGemmCacheDB()
{
    return gemm_cache_version_;
}

// Get norm cache version
int ProfileCacheDB::GetNormCacheDB()
{
    return norm_cache_version_;
}

// Delete an existing table in the db
// void DeleteExistTable(const std::string& table_kind)
// {
//     jinja2::Template tpl;
//     CheckTemplate(tpl.Load(g_query_all_table_template));
//     auto sql = query_all_table_tpl.RenderAsString({}).value();

//     std::vector<std::string> target_tables_;

//     CheckSqlite(sqlite3_exec(
//                     cache_db_,
//                     sql.c_str(),
//                     [&](void* para, int column_count, char** column_value, char** column_name) -> int {
//                         for (int i = 0; i < column_count; i++) {
//                             if (static_cast<std::string>(column_value[i])
//                                     .starts_with(fmt::format("{}_{}", device_name_, table_kind))) {
//                                 target_tables_.push_back(column_value[i]);
//                             }
//                         }
//                         return 0;
//                     },
//                     nullptr,
//                     nullptr),
//                 cache_db_);

//     if (!target_tables_.size()) {
//         throw std::runtime_error("no table kind exist")
//     }

//     if (target_tables_.size() != 1) {
//         throw std::runtime_error("only one table kind but got target_tables_")
//     }

//     CheckSqlite(sqlite3_exec(cache_db_, fmt::format("DROP TABLE {}", target_tables_[0])), cache_db_);
// }

// a function to query op from cache
std::tuple<std::string, int, int> ProfileCacheDB::Query(const std::string& sql)
{
    std::vector<std::string> query_algo;
    std::vector<int>         workspace_size;
    std::vector<int>         split_k;

    if (mode_ == CacheModeType::Local) {
        sqlite3_stmt* stmt;
        ATER_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, sql.c_str(), -1, &stmt, NULL), cache_db_);

        int column_count = 0;
        while (true) {
            auto step_result = sqlite3_step(stmt);  // SQLITE_DONE

            if (step_result != SQLITE_DONE && step_result != SQLITE_ROW) {
                break;
            }

            if (step_result == SQLITE_DONE && sqlite3_column_type(stmt, 0) == SQLITE_NULL) {
                return std::make_tuple("null", -1, -1);
            }

            query_algo.emplace_back(
                std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, column_count++))));
            workspace_size.emplace_back(sqlite3_column_int(stmt, column_count++));
            split_k.emplace_back(sqlite3_column_int(stmt, column_count++));
        };
        ATER_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    int min_workspace_size_idx =
        std::min_element(workspace_size.begin(), workspace_size.end()) - workspace_size.begin();
    return std::make_tuple(
        query_algo[min_workspace_size_idx], workspace_size[min_workspace_size_idx], split_k[min_workspace_size_idx]);

    // int (*callback)(void*, int, char**, char**) =
    //     LambdaToPointer([&](void* para, int column_count, char** column_value, char** column_name) -> int {
    //         query_algo.reserve(column_count);
    //         workspace_size.reserve(column_count);
    //         splitk_vec.reserve(column_count);

    //         for (int i = 0; i < column_count; i++) {
    //             query_algo.emplace_back(column_value[0]);
    //             workspace_size.emplace_back(std::atoi(column_value[1]));
    //             splitk_vec.emplace_back(std::atoi(column_value[2]));
    //         }
    //         return 0;
    //     });

    // if (mode_ == CacheModeType::Local) {
    //     ATER_CHECK_SQLITE(sqlite3_exec(cache_db_, sql.c_str(), callback, NULL, NULL), cache_db_);

    //     // if have many algos, choose one algo that has min workspace_size
    //     int min_workspace_size_idx =
    //         std::min_element(workspace_size.begin(), workspace_size.end()) - workspace_size.begin();
    //     return std::make_tuple(query_algo[min_workspace_size_idx],
    //                            workspace_size[min_workspace_size_idx],
    //                            splitk_vec[min_workspace_size_idx]);
    // }

    // else
    // {
    //     throw std::runtime_error("not implement");
    // }
}

// a function to query gemm op epilogue from cache
std::tuple<std::string, int, int>
ProfileCacheDB::QueryGemm(const std::unordered_map<std::string, std::variant<int, std::string>>& args)
{
    int gemm_curr_cache_version = GetGemmCacheVersion();
    LOG(INFO) << "using gemm_curr_cache_version: " << gemm_curr_cache_version;

    jinja2::ValuesMap gemm_query_value_map{
        {"dev", device_name_},
        {"version", gemm_curr_cache_version},
        {"dtype_a", std::get<int>(args.find("dtype_a")->second)},
        {"dtype_b", std::get<int>(args.find("dtype_b")->second)},
        {"dtype_c", std::get<int>(args.find("dtype_c")->second)},
        {"dtype_acc", std::get<int>(args.find("dtype_acc")->second)},
        {"major_a", std::get<int>(args.find("major_a")->second)},
        {"major_b", std::get<int>(args.find("major_b")->second)},
        {"major_c", std::get<int>(args.find("major_c")->second)},
        {"op_name", std::get<std::string>(args.find("op_name")->second)},
        {"device", device_name_},
        {"epilogue", std::get<int>(args.find("epilogue")->second)},
        {"pshape", std::get<std::string>(args.find("pshape")->second)},
        {"exec_entry_sha1", std::get<std::string>(args.find("exec_entry_sha1")->second)}};
    auto sql = TemplateLoadAndRender(g_gemm_query_source, gemm_query_value_map);
    return Query(sql);
}

// // a function to query normalization op epilogue from cache
// const int QueryNormalization()
// {

//     jinja2::Template  norm_tpl;
//     int               norm_curr_cache_version = GetNormCacheDB();
//     jinja2::ValuesMap norm_query_value_map{
//         {"dev", device_name_},
//         {"version", norm_curr_cache_version},
//         {"exec_entry", args.find("exec_entry")->second},
//         {"exec_entry_sha1", args.find("exec_entry_sha1")->second},
//         {"dtype_in", args.find("dtype_in")->second},
//         {"dtype_out", args.find("dtype_out")->second},
//         {"dtype_acc", args.find("dtype_acc")->second},
//         {"rank", args.find("rank")->second},
//         {"op_name", args.find("op_name")->second},
//         {"device", device_name_},
//     };

//     auto sql = norm_tpl.RenderAsString(norm_query_value_map).value();
//     return Query(sql);
// }

// a function to insert op into cache
int ProfileCacheDB::CheckIfInsert(const std::string& query_sql)
{
    int n_query_result = 0;

    if (mode_ == CacheModeType::Local) {
        sqlite3_stmt* stmt;
        ATER_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, query_sql.c_str(), -1, &stmt, nullptr), cache_db_);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            if (sqlite3_column_type(stmt, 0) == SQLITE_NULL) {
                break;
            }
            n_query_result += 1;
        };
        ATER_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    return n_query_result;
}

// if (mode_ == CacheModeType::Local) {
//     int (*callback)(void*, int, char**, char**) =
//         LambdaToPointer([&](void* para, int column_count, char** column_value, char** column_name) -> int {
//             n_query_result += 1;
//             return 0;
//         });
//     ATER_CHECK_SQLITE(sqlite3_exec(cache_db_, query_sql.c_str(), callback, NULL, NULL), cache_db_);

//     if (!n_query_result) {
//         ATER_CHECK_SQLITE(sqlite3_exec(cache_db_, insert_sql.c_str(), NULL, NULL, NULL), cache_db_);
//     }
//     else {
//         fmt::print("Ignore repeat profile_record:{query_sql}", fmt::arg("query_sql", query_sql));
//     }
// }

// else {
//     throw std::runtime_error("not implement");
// }

// a function to insert gemm op epilogue into cache
void ProfileCacheDB::InsertGemm(const std::unordered_map<std::string, std::variant<int, std::string>>& args)
{

    jinja2::ValuesMap gemm_query_value_map{
        {"dev", device_name_},
        {"version", gemm_cache_version_},
        {"dtype_a", std::get<int>(args.find("dtype_a")->second)},
        {"dtype_b", std::get<int>(args.find("dtype_b")->second)},
        {"dtype_c", std::get<int>(args.find("dtype_c")->second)},
        {"dtype_acc", std::get<int>(args.find("dtype_acc")->second)},
        {"major_a", std::get<int>(args.find("major_a")->second)},
        {"major_b", std::get<int>(args.find("major_b")->second)},
        {"major_c", std::get<int>(args.find("major_c")->second)},
        {"op_name", std::get<std::string>(args.find("op_name")->second)},
        {"device", device_name_},
        {"epilogue", std::get<int>(args.find("epilogue")->second)},
        {"pshape", std::get<std::string>(args.find("pshape")->second)},
        {"exec_entry_sha1", std::get<std::string>(args.find("exec_entry_sha1")->second)}};

    jinja2::ValuesMap gemm_insert_value_map{{"dev", device_name_}, {"version", gemm_cache_version_}};

    auto query_sql  = TemplateLoadAndRender(g_gemm_query_source, gemm_query_value_map);
    auto insert_sql = TemplateLoadAndRender(g_gemm_insert_source, gemm_insert_value_map);

    const int n_query_result = CheckIfInsert(query_sql);

    VLOG(1) << "query_sql: " << query_sql;
    VLOG(1) << "insert_sql: " << insert_sql;
    VLOG(1) << "n_query_result: " << n_query_result;

    if (!n_query_result) {
        sqlite3_stmt* stmt;
        ATER_CHECK_SQLITE(sqlite3_prepare_v2(cache_db_, insert_sql.c_str(), -1, &stmt, nullptr), cache_db_);

        int column_count = 1;

        ATER_CHECK_SQLITE(sqlite3_bind_text(stmt,
                                            column_count++,
                                            (std::get<std::string>(args.find("exec_entry")->second)).c_str(),
                                            -1,
                                            SQLITE_TRANSIENT),
                          cache_db_);

        ATER_CHECK_SQLITE(sqlite3_bind_text(stmt,
                                            column_count++,
                                            (std::get<std::string>(args.find("exec_entry_sha1")->second)).c_str(),
                                            -1,
                                            SQLITE_TRANSIENT),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("dtype_a")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("dtype_b")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("dtype_c")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("dtype_acc")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("major_a")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("major_b")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("major_c")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_text(stmt,
                                            column_count++,
                                            std::get<std::string>(args.find("op_name")->second).c_str(),
                                            -1,
                                            SQLITE_TRANSIENT),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("epilogue")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_text(stmt, column_count++, device_name_.c_str(), -1, SQLITE_TRANSIENT),
                          cache_db_);
        ATER_CHECK_SQLITE(
            sqlite3_bind_text(
                stmt, column_count++, (std::get<std::string>(args.find("algo")->second)).c_str(), -1, SQLITE_TRANSIENT),
            cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("workspace")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_int(stmt, column_count++, std::get<int>(args.find("split_k")->second)),
                          cache_db_);
        ATER_CHECK_SQLITE(sqlite3_bind_text(stmt,
                                            column_count++,
                                            (std::get<std::string>(args.find("pshape")->second)).c_str(),
                                            -1,
                                            SQLITE_TRANSIENT),
                          cache_db_);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            LOG(INFO) << "Insert gemm success";
        }
        else {
            LOG(ERROR) << "Insert failed";
        }

        ATER_CHECK_SQLITE(sqlite3_finalize(stmt), cache_db_);
    }

    else {
        LOG(WARNING) << "Ignore repeat profile_record:" << query_sql;
    }

    // auto insert_sql = TemplateLoadAndRender(g_gemm_insert_source, gemm_insert_value_map);
    // Insert(query_sql, insert_map);
}

// // a function to insert norm op into cache
// void InsertNorm(const std::map<std::string, std::any>& args)
// {
//     jinja2::Template norm_query_tpl, norm_insert_tpl;

//     jinja2::ValuesMap norm_query_value_map{{"dev", device_name_},
//                                            {"version", norm_cache_version_},
//                                            {"dtype_in", args.find("dtype_in")->second},
//                                            {"dtype_out", args.find("dtype_out")->second},
//                                            {"dtype_acc", args.find("dtype_acc")->second},
//                                            {"rank", args.find("rank")->second},
//                                            {"op_name", args.find("op_name")->second},
//                                            {"device", device_name_},
//                                            {"exec_entry_sha1", args.find("exec_entry_sha1")->second}};

//     jinja2::ValuesMap norm_insert_value_map{{"dev", device_name_},
//                                             {"version", norm_cache_version_},
//                                             {"exec_entry", args.find("exec_entry")->second},
//                                             {"exec_entry_sha1", args.find("exec_entry_sha1")->second},
//                                             {"dtype_in", args.find("dtype_in")->second},
//                                             {"dtype_out", args.find("dtype_out")->second},
//                                             {"dtype_acc", args.find("dtype_acc")->second},
//                                             {"rank", args.find("rank")->second},
//                                             {"op_name", args.find("op_name")->second},
//                                             {"device", device_name_},
//                                             {"algo", args.find("algo")->second},
//                                             {"workspace", args.find("workspace")->second}};

//     const std::string query_sql  = norm_query_tpl.RenderAsString(norm_query_value_map).value();
//     const std::string insert_sql = norm_insert_tpl.RenderAsString(norm_insert_value_map).value();

//     Insert(query_sql, insert_sql);
// }

ProfileCacheDB::~ProfileCacheDB()
{
    ATER_CHECK_SQLITE(sqlite3_close_v2(cache_db_), cache_db_);
}

}  // namespace ater