#pragma once

#include <filesystem>
#include <memory>
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>

#include "flashck/core/profiling/codegen_utils.h"

namespace flashck {

#define CHECK_SQLITE3(expr, db)                                                                                        \
    do {                                                                                                               \
        int result_code = (expr);                                                                                      \
        if (result_code != SQLITE_OK) {                                                                                \
            const char* err = sqlite3_errmsg(db);                                                                      \
            throw std::runtime_error("SQLite error[" + std::to_string(result_code)                                     \
                                     + "]: " + (err ? err : "unknown error") + " at " + std::string(__FILE__) + ":"    \
                                     + std::to_string(__LINE__));                                                      \
        }                                                                                                              \
    } while (0)

#define CHECK_SQLITE3_RC(expr, db, rc)                                                                                 \
    do {                                                                                                               \
        rc = (expr);                                                                                                   \
        if (rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE) {                                                \
            const char* err = sqlite3_errmsg(db);                                                                      \
            throw std::runtime_error("SQLite error[" + std::to_string(rc) + "]: " + (err ? err : "unknown error")      \
                                     + " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));               \
        }                                                                                                              \
    } while (0)

class StmtWrapper {
public:
    explicit StmtWrapper(sqlite3* db, const char* sql):
        stmt_(
            [db, sql] {
                sqlite3_stmt* stmt = nullptr;
                CHECK_SQLITE3(sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr), db);
                return stmt;
            }(),
            &sqlite3_finalize)
    {
    }

    operator sqlite3_stmt*() const
    {
        return stmt_.get();
    }

private:
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt_;
};

class ProfilingDB {
public:
    explicit ProfilingDB(const std::filesystem::path& path);

    // Query the database for norm profiling information(algo, split_k, latency, tflops, bandwidth)
    std::tuple<std::string, PerfResult> Query(InstanceData& instance_data);

    // Check if an entry exists in the database
    bool CheckIfInsert(InstanceData& instance_data);

    // Insert a new entry into the database
    void Insert(InstanceData& instance_data);

private:
    void Execute(const char* sql)
    {
        CHECK_SQLITE3(sqlite3_exec(db_ptr_.get(), sql, nullptr, nullptr, nullptr), db_ptr_.get());
    }

    std::unique_ptr<sqlite3, decltype(&sqlite3_close_v2)> db_ptr_;

    std::filesystem::path path_;
};

}  // namespace flashck