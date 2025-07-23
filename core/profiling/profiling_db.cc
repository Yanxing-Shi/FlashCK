#include "core/profiling/profiling_db.h"

namespace flashck {

ProfilingDB::ProfilingDB(const std::filesystem::path& path):
    db_ptr_(
        [path] {
            sqlite3* raw_db_ptr = nullptr;
            // Open database with read-write access, create if doesn't exist
            CHECK_SQLITE3(sqlite3_open_v2(
                              path.string().c_str(), &raw_db_ptr, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr),
                          raw_db_ptr);
            return raw_db_ptr;
        }(),
        &sqlite3_close),
    path_(path)
{
    try {
        VLOG(1) << "Initializing profiling database at: " << path_.string();

        // Configure SQLite for optimal performance and reliability
        Execute("PRAGMA journal_mode = WAL");    // Write-Ahead Logging for better concurrency
        Execute("PRAGMA synchronous = NORMAL");  // Balanced safety and performance
        Execute("PRAGMA foreign_keys = ON");     // Enable referential integrity

        // Create database schema with comprehensive constraints
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

        // Execute schema creation
        for (const char* sql : {schema}) {
            Execute(sql);
        }

        VLOG(1) << "Database schema initialized successfully";
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Failed to initialize database: " << e.what();
        throw;
    }
    catch (...) {
        LOG(ERROR) << "Unknown error during database initialization";
        throw;
    }
}

std::tuple<std::string, PerfResult> ProfilingDB::Query(InstanceData& instance_data)
{
    VLOG(2) << "Querying database for instance: " << instance_data.instance_name_;

    // SQL template for querying optimal instance by environment and problem parameters
    constexpr const char* sql = R"sql(
        SELECT instance_name, split_k, latency, tflops, bandwidth FROM {type}
        WHERE rocm_version=?
        AND device_name=?
        AND setting=?
        AND problem=?
        LIMIT 1
    )sql";

    // Format SQL with appropriate table name based on code generation kind
    std::string formatted_sql = Sprintf(sql, fmt::arg("type", CodeGenKindToString(instance_data.code_gen_kind_)));

    StmtWrapper   stmt(db_ptr_.get(), formatted_sql.c_str());
    sqlite3_stmt* raw_stmt = stmt;

    // Bind query parameters in order
    int idx = 1;
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.rocm_version_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.device_name_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(sqlite3_bind_text(stmt, idx++, instance_data.setting_.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                  db_ptr_.get());

    // Bind problem data using visitor pattern to handle different problem types
    instance_data.VisitProblem([&](auto&& problem) {
        CHECK_SQLITE3(sqlite3_bind_text(stmt, idx++, problem.Serialize().c_str(), -1, SQLITE_TRANSIENT), db_ptr_.get());
    });

    // Execute query and process results
    int rc;
    CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);

    if (rc == SQLITE_ROW) {
        // Extract results from database row
        std::string instance_name = reinterpret_cast<const char*>(sqlite3_column_text(raw_stmt, 0));
        PerfResult  perf_result{
            sqlite3_column_int(raw_stmt, 1),     // split_k
            sqlite3_column_double(raw_stmt, 2),  // latency
            sqlite3_column_double(raw_stmt, 3),  // tflops
            sqlite3_column_double(raw_stmt, 4)   // bandwidth
        };

        VLOG(1) << "Found cached instance: " << instance_name << " with latency=" << perf_result.latency_ << "ms, "
                << "tflops=" << perf_result.tflops_ << ", "
                << "bandwidth=" << perf_result.bandwidth_ << "GB/s";

        // Clean up statement state
        CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

        return std::make_tuple(instance_name, perf_result);
    }
    else if (rc != SQLITE_DONE) {
        // Handle unexpected SQL errors
        std::string error_msg = sqlite3_errmsg(db_ptr_.get());
        LOG(ERROR) << "Database query failed: " << error_msg;
        throw std::runtime_error("Database query error: " + error_msg);
    }

    // Clean up statement state for no-match case
    CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
    CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

    VLOG(2) << "No cached instance found for query";
    return std::make_tuple("", PerfResult{-1, -1.0f, -1.0f, -1.0f});
}

bool ProfilingDB::CheckIfInsert(InstanceData& instance_data)
{
    VLOG(2) << "Checking existence for instance: " << instance_data.instance_name_;

    // SQL template for existence check (returns 1 if record exists)
    constexpr const char* sql = R"sql(
    SELECT 1 FROM {type}
    WHERE rocm_version=? 
    AND device_name=?
    AND setting=?
    AND problem=?
    LIMIT 1
)sql";

    // Format SQL with appropriate table name
    std::string formatted_sql = Sprintf(sql, fmt::arg("type", CodeGenKindToString(instance_data.code_gen_kind_)));

    StmtWrapper   stmt(db_ptr_.get(), formatted_sql.c_str());
    sqlite3_stmt* raw_stmt = stmt;

    // Bind parameters for existence check
    int idx = 1;
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.rocm_version_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(
        sqlite3_bind_text(raw_stmt, idx++, instance_data.environment_.device_name_.c_str(), -1, SQLITE_TRANSIENT),
        db_ptr_.get());
    CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, instance_data.setting_.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                  db_ptr_.get());

    // Bind problem data
    instance_data.VisitProblem([&](auto&& problem) {
        CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, problem.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                      db_ptr_.get());
    });

    // Execute existence check
    int rc;
    CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);

    // Clean up statement state
    CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
    CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

    bool exists = (rc == SQLITE_ROW);
    if (rc == SQLITE_DONE) {
        VLOG(2) << "No matching records found for existence check";
    }
    else if (exists) {
        VLOG(2) << "Record already exists in database";
    }

    return exists;
}

void ProfilingDB::Insert(InstanceData& instance_data)
{
    VLOG(1) << "Attempting to insert instance: " << instance_data.instance_name_;

    // Check for duplicate entries before insertion
    if (CheckIfInsert(instance_data)) {
        LOG(WARNING) << "Record already exists, skipping insertion for: " << instance_data.instance_name_;
        return;
    }

    // Use transaction for atomicity
    Execute("BEGIN TRANSACTION");
    try {
        // SQL template for inserting new optimal instance
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

        // Format SQL with appropriate table name
        std::string formatted_sql = Sprintf(sql, fmt::arg("type", CodeGenKindToString(instance_data.code_gen_kind_)));

        StmtWrapper   stmt(db_ptr_.get(), formatted_sql.c_str());
        sqlite3_stmt* raw_stmt = stmt;

        // Bind all insertion parameters
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

        // Bind problem data using visitor pattern
        instance_data.VisitProblem([&](auto&& problem) {
            CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, problem.Serialize().c_str(), -1, SQLITE_TRANSIENT),
                          db_ptr_.get());
        });

        // Bind instance metadata and performance metrics
        CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, instance_data.instance_name_.c_str(), -1, SQLITE_TRANSIENT),
                      db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_int(raw_stmt, idx++, instance_data.perf_result_.split_k_), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, instance_data.perf_result_.latency_), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, instance_data.perf_result_.tflops_), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, instance_data.perf_result_.bandwidth_), db_ptr_.get());

        // Execute insertion
        int rc;
        CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);

        // Clean up statement state
        CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

        // Commit transaction on success
        Execute("COMMIT");

        VLOG(1) << "Successfully inserted instance: " << instance_data.instance_name_
                << " with performance metrics: latency=" << instance_data.perf_result_.latency_ << "ms, "
                << "tflops=" << instance_data.perf_result_.tflops_ << ", "
                << "bandwidth=" << instance_data.perf_result_.bandwidth_ << "GB/s";
    }
    catch (const std::exception& e) {
        // Rollback transaction on any error
        Execute("ROLLBACK");
        LOG(ERROR) << "Failed to insert instance data: " << e.what();
        throw;
    }
    catch (...) {
        // Rollback on unknown errors
        Execute("ROLLBACK");
        LOG(ERROR) << "Unknown error during instance insertion";
        throw;
    }
}

}  // namespace flashck