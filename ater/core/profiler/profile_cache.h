#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <variant>

#include <sqlite3.h>

/*
SQLite backend for conv/gemm profiling cache
*/

namespace ater {

const static std::string g_check_table_exist_source = R"(
SELECT name FROM sqlite_master WHERE type='table' AND name='{{table_name}}';
)";

const static std::string g_query_all_table_source = R"(
SELECT name FROM sqlite_master WHERE type='table';
)";

const static std::string g_gemm_init_source = R"(
CREATE TABLE IF NOT EXISTS {{dev}}_gemm_{{version}} (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192),
  exec_entry_sha1 VARCHAR(64),
  dtype_a INTEGER,
  dtype_b INTEGER,
  dtype_c INTEGER,
  dtype_acc INTEGER,
  major_a INTEGER,
  major_b INTEGER,
  major_c INTEGER,
  op_name VARCHAR(512),
  epilogue VARCHAR(512),
  device VARCHAR(16),
  algo VARCHAR(512),
  workspace INTEGER DEFAULT 0,
  duration FLOAT DEFAULT -1,
  split_k INTEGER DEFAULT 1,
  pshape VARCHAR(64)
);
)";

// const static std::string g_norm_init_source = R"(
// CREATE TABLE IF NOT EXISTS {{dev}}_normalization_{{version}} (
//   id INTEGER PRIMARY KEY AUTOINCREMENT,
//   exec_entry VARCHAR(8192) NOT NULL,
//   exec_entry_sha1 VARCHAR(64) NOT NULL,
//   dtype_in INTEGER NOT NULL,
//   dtype_acc INTEGER NOT NULL,
//   dtype_out INTEGER NOT NULL,
//   rank INTEGER NOT NULL,
//   op_name VARCHAR(512) NOT NULL,
//   device VARCHAR(16) NOT NULL,
//   algo VARCHAR(512) NOT NULL,
//   workspace INTEGER DEFAULT 0,
//   duration FLOAT DEFAULT -1,
//   template_ver INTEGER NOT NULL DEFAULT 290,
//   created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
// );
// )";

const static std::string g_gemm_query_source = R"(
SELECT algo, workspace, split_k
FROM {{dev}}_gemm_{{version}}
WHERE
dtype_a={{dtype_a}} AND
dtype_b={{dtype_b}} AND
dtype_c={{dtype_c}} AND
dtype_acc={{dtype_acc}} AND
major_a={{major_a}} AND
major_b={{major_b}} AND
major_c={{major_c}} AND
op_name='{{op_name}}' AND
device='{{device}}' AND
epilogue={{epilogue}} AND
pshape='{{pshape}}' AND
exec_entry_sha1='{{exec_entry_sha1}}';
)";

// const static std::string g_norm_query_source = R"(
// SELECT algo, workspace
// FROM {{dev}}_normalization_{{version}}
// WHERE
// dtype_in={{dtype_in}} AND
// dtype_out={{dtype_out}} AND
// dtype_acc={{dtype_acc}} AND
// rank={{rank}} AND
// op_name='{{op_name}}' AND
// device='{{device}}' AND
// exec_entry_sha1='{{exec_entry_sha1}}';
// )";

const static std::string g_gemm_insert_source = R"(
INSERT INTO {{dev}}_gemm_{{version}} (
    exec_entry,
    exec_entry_sha1,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    major_a,
    major_b,
    major_c,
    op_name,
    epilogue,
    device,
    algo,
    workspace,
    split_k,
    pshape
)
VALUES (
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?
);
)";

// const static std::string g_norm_insert_source = R"(
// INSERT INTO {{dev}}_normalization_{{version}} (
//     exec_entry,
//     exec_entry_sha1,
//     dtype_in,
//     dtype_out,
//     dtype_acc,
//     rank,
//     op_name,
//     device,
//     algo,
//     workspace
// )
// VALUES (
//     '{{exec_entry}}',
//     '{{exec_entry_sha1}}',
//     {{dtype_in}},
//     {{dtype_out}},
//     {{dtype_acc}},
//     {{rank}},
//     '{{op_name}}',
//     '{{device}}',
//     '{{algo}}',
//     {{workspace}}
// );
// )";

enum class CacheModeType {
    Local  = 0,
    Remote = 1,
};

/*
Local SQLite profile cache database.
*/

class ProfileCacheDB {
public:
    /*
    Parameters
        ----------
        target : str
            target device name. CUDA or ROCM
        path : str, optional
            path to the database file. If not specified, a temporary file is created.
        uri : str, optional
            uri to the RESFul API (Not implemented yet)
        port : str, optional
            port to the RESFul API (Not implemented yet)
    */
    explicit ProfileCacheDB(const std::string&           device_name,
                            const std::filesystem::path& path,
                            const std::string&           uri  = "",
                            const std::string&           port = "");

    // Gemm_cache_version
    int GetGemmCacheVersion();
    // norm_cache_version
    int GetNormCacheVersion();

    // Creates gemm table
    void CreateGemmTable();

    bool TableExists(const std::string& table_kind, const int table_version);

    // Get gemm cache version
    int GetGemmCacheDB();

    // Get norm cache version
    int GetNormCacheDB();

    // a function to query op from cache
    std::tuple<std::string, int, int> Query(const std::string& sql);

    // a function to query gemm op epilogue from cache
    std::tuple<std::string, int, int>
    QueryGemm(const std::unordered_map<std::string, std::variant<int, std::string>>& args);

    // a function to insert op into cache
    int CheckIfInsert(const std::string& query_sql);

    // a function to insert gemm op epilogue into cache
    void InsertGemm(const std::unordered_map<std::string, std::variant<int, std::string>>& args);

    ~ProfileCacheDB();

private:
    sqlite3*                    cache_db_ = nullptr;
    const std::string           device_name_;
    const std::filesystem::path path_;
    const std::string           uri_;
    const std::string           port_;

    CacheModeType mode_;

    int gemm_cache_version_;
    int norm_cache_version_;
};

}  // namespace ater