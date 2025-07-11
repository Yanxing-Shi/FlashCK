#include "flashck/core/profiling/profiling_db.h"

namespace flashck {

ProfilingDB::ProfilingDB(const std::filesystem::path& path):
    db_ptr_(
        [path] {
            sqlite3* raw_db_ptr = nullptr;
            CHECK_SQLITE3(sqlite3_open_v2(
                              path.string().c_str(), &raw_db_ptr, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr),
                          raw_db_ptr);
            return raw_db_ptr;
        }(),
        &sqlite3_close_v2),
    path_(path)
{
    try {
        Execute("PRAGMA journal_mode = WAL");
        Execute("PRAGMA synchronous = NORMAL");
        Execute("PRAGMA foreign_keys = ON");

        constexpr const char* schema = R"sql(
                CREATE TABLE IF NOT EXISTS norm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rocm_version TEXT NOT NULL CHECK(length(rocm_version) > 0),
                    device_name TEXT NOT NULL CHECK(length(device_name) > 0),
                    setting TEXT NOT NULL CHECK(json_valid(setting)),
                    problem TEXT NOT NULL CHECK(json_valid(problem)),
                    instance_name TEXT NOT NULL CHECK(length(instance_name) > 0),
                    split_k INTEGER NOT NULL DEFAULT -1 CHECK(split_k >= -1),
                    latency REAL CHECK(latency > 0),
                    tflops REAL CHECK(tflops > 0),
                    bandwidth REAL CHECK(bandwidth > 0)
                );
            )sql";

        for (const char* sql : {schema}) {
            Execute(sql);
        }
    }
    catch (...) {
        throw;
    }
}

// Query the database for norm profiling information(algo, split_k, latency, tflops, bandwidth)
std::tuple<std::string, PerfResult> ProfilingDB::Query(InstanceData& instance_data)
{
    constexpr const char* sql = R"sql(
        SELECT instance_name, split_k, latency, tflops, bandwidth FROM {type}
        WHERE rocm_version=?
        AND device_name=?
        AND setting=?
        AND problem=?
        LIMIT 1
    )sql";

    std::string formatted_sql = Sprintf(sql, fmt::arg("type", CodeGenKindToString(instance_data.code_gen_kind_)));

    StmtWrapper   stmt(db_ptr_.get(), formatted_sql.c_str());
    sqlite3_stmt* raw_stmt = stmt;

    int idx = 1;
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.rocm_version_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.device_name_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(sqlite3_bind_text(stmt, idx++, instance_data.setting_.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                  db_ptr_.get());

    instance_data.VisitProblem([&](auto&& problem) {
        CHECK_SQLITE3(sqlite3_bind_text(stmt, idx++, problem.Serialize().c_str(), -1, SQLITE_TRANSIENT), db_ptr_.get());
    });

    int rc;
    CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);

    if (rc == SQLITE_ROW) {
        return std::make_tuple(reinterpret_cast<const char*>(sqlite3_column_text(raw_stmt, 0)),
                               PerfResult{sqlite3_column_int(raw_stmt, 1),
                                          sqlite3_column_double(raw_stmt, 2),
                                          sqlite3_column_double(raw_stmt, 3),
                                          sqlite3_column_double(raw_stmt, 4)});
    }
    else if (rc != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db_ptr_.get()));
    }
    CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
    CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

    return std::make_tuple("", PerfResult{-1, -1.0f, -1.0f, -1.0f});
}

// Check if an entry exists in the database
bool ProfilingDB::CheckIfInsert(InstanceData& instance_data)
{
    constexpr const char* sql = R"sql(
    SELECT 1 FROM {type}
    WHERE rocm_version=? 
    AND device_name=?
    AND setting=?
    AND problem=?
    LIMIT 1
)sql";

    std::string formatted_sql = Sprintf(sql, fmt::arg("type", CodeGenKindToString(instance_data.code_gen_kind_)));

    StmtWrapper   stmt(db_ptr_.get(), formatted_sql.c_str());
    sqlite3_stmt* raw_stmt = stmt;

    int idx = 1;
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.rocm_version_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.device_name_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, instance_data.setting_.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                  db_ptr_.get());
    instance_data.VisitProblem([&](auto&& problem) {
        CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, problem.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                      db_ptr_.get());
    });

    int rc;
    CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);
    CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
    CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

    if (rc == SQLITE_DONE) {
        std::cout << "No matching records found" << std::endl;
    }
    return (rc == SQLITE_ROW);
}

// Insert a new entry into the database
void ProfilingDB::Insert(InstanceData& instance_data)
{
    Execute("BEGIN TRANSACTION");
    try {
        constexpr const char* sql = R"sql(
            INSERT INTO {type}
                (rocm_version, 
                 device_name,
                 setting,
                 problem, 
                 instance_name,
                 split_k,
                 latency, 
                 tflops, 
                 bandwidth)
            VALUES (?1, 
                    ?2,     
                    ?3, 
                    ?4, 
                    ?5, 
                    ?6, 
                    ?7, 
                    ?8, 
                    ?9)
        )sql";

        std::string formatted_sql = Sprintf(sql, fmt::arg("type", CodeGenKindToString(instance_data.code_gen_kind_)));

        StmtWrapper   stmt(db_ptr_.get(), formatted_sql.c_str());
        sqlite3_stmt* raw_stmt = stmt;

        int idx = 1;
        CHECK_SQLITE3(
            sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.rocm_version_.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.device_name_.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(raw_stmt, idx++, instance_data.setting_.Serialize().c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        instance_data.VisitProblem([&](auto&& problem) {
            CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, problem.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                          db_ptr_.get());
        });
        CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, instance_data.instance_name_.c_str(), -1, SQLITE_TRANSIENT),
                      db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_int(raw_stmt, idx++, instance_data.perf_result_.split_k_), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, instance_data.perf_result_.latency_), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, instance_data.perf_result_.tflops_), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, instance_data.perf_result_.bandwidth_), db_ptr_.get());

        int rc;
        CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);
        CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

        Execute("COMMIT");
    }
    catch (...) {
        Execute("ROLLBACK");
        throw;
    }
}

}  // namespace flashck