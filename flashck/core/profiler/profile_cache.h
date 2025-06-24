#pragma once

#include "flashck/core/profiler/embedding_cache_entry.h"
#include "flashck/core/profiler/fmha_cache_entry.h"
#include "flashck/core/profiler/gemm_cache_entry.h"
#include "flashck/core/profiler/library.h"
#include "flashck/core/profiler/norm_cache_entry.h"

extern "C" {
#include <sqlite3.h>
}

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>

namespace flashck {

inline constexpr std::string_view kCheckTableExistSource = R"sql(
SELECT name FROM sqlite_master WHERE type='table' AND name='{{table_name}}';
)sql";

inline constexpr std::string_view kGemmInitSource = R"sql(
CREATE TABLE IF NOT EXISTS {{device_name}}_gemm (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192),
  exec_entry_sha1 VARCHAR(64),
  a_dtype VARCHAR(8),
  b_dtype VARCHAR(8),
  c_dtype VARCHAR(8),
  acc_dtype VARCHAR(8),
  layout VARCHAR(8),
  op_kind VARCHAR(512),
  epilogue VARCHAR(512),
  algo VARCHAR(512),
  duration FLOAT DEFAULT -1,
  split_k INTEGER DEFAULT 1,
  permute_shape VARCHAR(64)
);
)sql";

inline constexpr std::string_view kNormInitSource = R"sql(
CREATE TABLE IF NOT EXISTS {{device_name}}_norm (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192),
  exec_entry_sha1 VARCHAR(64),
  x_dtype VARCHAR(8),
  y_dtype VARCHAR(8),
  smooth_scale_dtype VARCHAR(8),
  y_scale_dtype VARCHAR(8),
  op_kind VARCHAR(512),
  epilogue VARCHAR(512),
  algo VARCHAR(512),
  fused_add VARCHAR(512),
  fused_quant VARCHAR(512),
  duration FLOAT DEFAULT -1
);
)sql";

inline constexpr std::string_view kFmhaInitSource = R"sql(
CREATE TABLE IF NOT EXISTS {{device_name}}_fmha (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192),
  exec_entry_sha1 VARCHAR(64),
  dtype VARCHAR(8),
  mask_type VARCHAR(8),
  bias_type VARCHAR(8),
  mode VARCHAR(8),
  rotary_dim INTEGER,
  paged_block_size INTEGER,
  is_use_batch_cache_idx INTEGER,
  op_kind VARCHAR(512),
  epilogue VARCHAR(512),
  algo VARCHAR(512),
  duration FLOAT DEFAULT -1,
  num_splits INTEGER DEFAULT 1
);
)sql";

inline constexpr std::string_view kEmbeddingInitSource = R"sql(
CREATE TABLE IF NOT EXISTS {{device_name}}_embedding (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192),
  exec_entry_sha1 VARCHAR(64),
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
  epilogue VARCHAR(512),
  algo VARCHAR(512),
  duration FLOAT DEFAULT -1
);
)sql";

inline constexpr std::string_view kGemmQuerySource = R"sql(
SELECT algo, split_k
FROM {{device_name}}_gemm
WHERE
a_dtype='{{a_dtype}}' AND
b_dtype='{{b_dtype}}' AND
c_dtype='{{c_dtype}}' AND
acc_dtype='{{acc_dtype}}' AND
layout='{{layout}}' AND
op_kind='{{op_kind}}' AND
epilogue='{{epilogue}}' AND
permute_shape='{{pshape}}' AND
exec_entry_sha1='{{exec_entry_sha1}}';
)sql";

inline constexpr std::string_view kNormQuerySource = R"sql(
SELECT algo
FROM {{device_name}}_norm
WHERE
x_dtype='{{x_dtype}}' AND
y_dtype='{{y_dtype}}' AND
smooth_scale_dtype='{{smooth_scale_dtype}}' AND
y_scale_dtype='{{y_scale_dtype}}' AND
op_name='{{op_name}}' AND
epilogue='{{epilogue}}' AND
exec_entry_sha1='{{exec_entry_sha1}}' AND
fused_add = '{{fused_add}}' AND
fused_quant = '{{fused_quant}}';
)sql";

inline constexpr std::string_view kFmhaQuerySource = R"sql(
SELECT algo
FROM {{device_name}}_fmha
WHERE
dtype='{{dtype}}' AND
mask_type='{{mask_type}}' AND
bias_type='{{bias_type}}' AND
mode='{{mode}}' AND
rotary_dim='{{rotary_dim}}' AND
paged_block_size='{{paged_block_size}}' AND
is_use_batch_cache_idx='{{is_use_batch_cache_idx}}' AND
op_name='{{op_name}}' AND
epilogue='{{epilogue}}' AND
exec_entry_sha1='{{exec_entry_sha1}}';
)sql";

inline constexpr std::string_view kEmbeddingQuerySource = R"sql(
SELECT algo
FROM {{device_name}}_embedding
WHERE
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
op_name='{{op_name}}' AND
epilogue='{{epilogue}}' AND
exec_entry_sha1='{{exec_entry_sha1}}';
)sql";

inline constexpr std::string_view kNormInsertSource = R"sql(
INSERT INTO {{device_name}}_norm (
    x_dtype,
    y_dtype,
    smooth_scale_dtype,
    y_scale_dtype,
    op_name,
    device,
    epilogue,
    exec_entry,
    exec_entry_sha1,
    fused_add,
    fused_quant,
    algo
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
    ?
);
)sql";

inline constexpr std::string_view kFmhaInsertSource = R"sql(
INSERT INTO {{device_name}}_fmha (
    dtype,
    mask,
    bias,
    mode,
    rotary_dim,
    paged_block_size,
    use_batch_cache_idx,
    op_name,
    epilogue,
    exec_entry,
    exec_entry_sha1,
    num_splits,
    algo
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
    ?
);
)sql";

inline constexpr std::string_view kEmbeddingInsertSource = R"sql(
INSERT INTO {{device_name}}_embedding (
    exec_entry,
    exec_entry_sha1,
    vocab_size,
    num_embeddings,
    type_vocab_size,
    max_position_embeddings,
    embedding_dims,
    emb_dtype,
    index_dtype,
    gamma_dtype,
    beta_dtype,
    acc_dtype,
    out_dtype,
    op_name,
    epilogue,
    algo
)
VALUES(
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
)sql";

// Defines where cached data is stored.
enum class CacheMode {
    kLocal  = 0,  // Data resides in local memory.
    kRemote = 1,  // Data is retrieved from remote storage (network-bound).
};

// Local SQLite profile cache database.
class ProfileCacheDB {
public:
    struct AutoStmt {
        sqlite3_stmt* stmt;
        sqlite3*      db;
        AutoStmt(sqlite3* db, const std::string& sql): db(db)
        {
            CHECK_SQLITE(sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr), db);
        }
        ~AutoStmt()
        {
            CHECK_SQLITE(sqlite3_finalize(stmt), db);
        }
    } stmt(cache_db_, sql);

    struct SQLiteDeleter {
        void operator()(sqlite3* db) const
        {
            if (db)
                sqlite3_close_v2(db);
        }
    };

    using SQLitePtr = std::unique_ptr<sqlite3, SQLiteDeleter>;

    explicit ProfileCacheDB(const std::string&           device_name,
                            const std::filesystem::path& path,
                            const std::string&           uri  = "",
                            const std::string&           port = "");

    // Creates gemm table
    void CreateGemmTable();

    // Creates norm table
    void CreateNormTable();

    // Creates fmha table
    void CreateFmhaTable();

    // Creates embedding table
    void CreateEmbeddingTable();

    bool TableExists(const GenOperationKind& op_kind);

    // a function to query op from cache
    std::tuple<std::string, int64_t> Query(const std::string& sql);

    // a function to query gemm op epilogue from cache
    std::tuple<std::string, int64_t> QueryGemm(const GemmQueryEntry& query);

    // a function to query norm op epilogue from cache
    std::tuple<std::string, int64_t> QueryNorm(const NormQueryEntry& query);

    // a function to query fmha op epilogue from cache
    std::tuple<std::string, int64_t> QueryFmha(const FmhaQueryEntry& query);

    // a function to query embedding op epilogue from cache
    std::tuple<std::string, int64_t> QueryEmbedding(const EmbeddingQueryEntry& query);

    // a function to insert op into cache
    int64_t CheckIfInsert(const std::string& query_sql);

    // a function to insert gemm op epilogue into cache
    void InsertGemm(const GemmRecordEntry& record);

    // a function to insert norm op epilogue into cache
    void InsertNorm(const NormRecordEntry& record);

    // a function to fmha epilogue into cache
    void InsertFmha(const FmhaRecordEntry& record);

    // a function to insert embedding op epilogue into cache
    void InsertEmbedding(const EmbeddingRecordEntry& record);

    ~ProfileCacheDB();

private:
    SQLitePtr cache_db_;

    const std::string           device_name_;
    const std::filesystem::path path_;
    const std::string           uri_;
    const std::string           port_;

    CacheMode mode_;
};

}  // namespace flashck