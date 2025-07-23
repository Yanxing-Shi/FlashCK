#pragma once

#include "core/profiling/profiling_helper.h"

namespace flashck {

/**
 * @class StmtWrapper
 * @brief RAII wrapper for SQLite prepared statements
 *
 * Provides automatic resource management for SQLite prepared statements,
 * ensuring proper cleanup through RAII principles. The wrapper automatically
 * prepares the statement on construction and finalizes it on destruction.
 */
class StmtWrapper {
public:
    /**
     * @brief Construct and prepare SQLite statement
     * @param db SQLite database connection handle
     * @param sql SQL statement string to prepare
     *
     * Automatically prepares the SQL statement and stores it with proper
     * resource management. Throws on preparation failure.
     */
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

    /**
     * @brief Implicit conversion to raw SQLite statement
     * @return Raw sqlite3_stmt pointer for use with SQLite API
     */
    operator sqlite3_stmt*() const
    {
        return stmt_.get();
    }

private:
    /// Managed SQLite statement with automatic finalization
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt_;
};

/**
 * @class ProfilingDB
 * @brief Database interface for kernel profiling result storage and retrieval
 *
 * Manages a SQLite database containing optimal kernel configurations for
 * different problem types and hardware environments. Supports querying
 * cached results to avoid redundant profiling and inserting new optimal
 * configurations discovered during profiling runs.
 *
 * Database Schema:
 * - Tables per code generation kind (norm, gemm, etc.)
 * - Environment identification (ROCm version, device)
 * - Problem parameters (serialized JSON)
 * - Performance metrics (latency, throughput, bandwidth)
 * - Instance metadata (name, split-K configuration)
 */
class ProfilingDB {
public:
    /**
     * @brief Initialize database connection and schema
     * @param path Filesystem path to SQLite database file
     *
     * Opens or creates the SQLite database with proper performance settings:
     * - WAL journal mode for better concurrency
     * - NORMAL synchronous mode for balanced safety/performance
     * - Foreign key constraints enabled
     * - Creates schema tables if they don't exist
     */
    explicit ProfilingDB(const std::filesystem::path& path);

    /**
     * @brief Query database for cached optimal instance
     * @param instance_data Instance data containing query parameters
     * @return Tuple of (instance_name, performance_result) or empty if not found
     *
     * Searches the database for a cached optimal instance matching the
     * environment, problem parameters, and settings. Returns the instance
     * name and performance metrics if found, or empty values if no match exists.
     *
     * Query parameters:
     * - ROCm version and device name
     * - Serialized problem and setting configurations
     * - Code generation kind (determines table)
     */
    std::tuple<std::string, PerfResult> Query(InstanceData& instance_data);

    /**
     * @brief Check if instance configuration already exists in database
     * @param instance_data Instance data to check for existence
     * @return True if matching record exists, false otherwise
     *
     * Performs existence check before insertion to prevent duplicate entries.
     * Uses the same query parameters as Query() but only checks for presence.
     */
    bool CheckIfInsert(InstanceData& instance_data);

    /**
     * @brief Insert new optimal instance into database
     * @param instance_data Complete instance data including performance metrics
     *
     * Inserts a new optimal kernel configuration into the database after
     * checking for duplicates. Uses transactions to ensure atomicity and
     * includes all relevant parameters for future cache lookups.
     *
     * Inserted data:
     * - Environment (ROCm version, device)
     * - Problem and setting configurations (JSON)
     * - Instance name and split-K value
     * - Performance metrics (latency, tflops, bandwidth)
     */
    void Insert(InstanceData& instance_data);

private:
    /**
     * @brief Execute SQL statement without parameters
     * @param sql SQL statement string to execute
     *
     * Utility method for executing simple SQL statements like pragma
     * settings, schema creation, and transaction control commands.
     */
    void Execute(const char* sql)
    {
        CHECK_SQLITE3(sqlite3_exec(db_ptr_.get(), sql, nullptr, nullptr, nullptr), db_ptr_.get());
    }

    /// Managed SQLite database connection with automatic closing
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_ptr_;

    /// Database file path for reference and error reporting
    std::filesystem::path path_;
};

}  // namespace flashck