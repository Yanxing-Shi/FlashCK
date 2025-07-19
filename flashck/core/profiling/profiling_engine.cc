#include "flashck/core/profiling/profiling_engine.h"

#include "flashck/core/utils/common.h"

FC_DECLARE_string(FC_HOME_PATH);
FC_DECLARE_string(FC_TUNING_DB_DIR);
FC_DECLARE_bool(FC_FLUSH_PROFILING_DB);

namespace flashck {

ProfilingEngine::ProfilingEngine()
{
    LOG(INFO) << "Initializing Profiling Engine on device: " << GetDeviceName();

    // Initialize graph code generation orchestrator
    try {
        graph_codegen_ = std::make_unique<GraphCodeGen>();
        LOG(INFO) << "Graph code generator initialized successfully";
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Graph code generator initialization failed: " << e.what();
        throw;
    }

    // Initialize profiling database with fallback mechanisms
    try {
        LoadProfilingDB();
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Profiling database initialization failed: " << e.what();
        throw;
    }
}

ProfilingEngine::~ProfilingEngine()
{
    // Resource cleanup is handled automatically by smart pointers
    // for profiling_db_, graph_codegen_, and kernel_lib_

    // Optional filesystem cleanup for profiling data
    // This section demonstrates how to implement cleanup if needed
    // Uncomment and configure if file cleanup is desired

    // Early exit if profiling cleanup is disabled via feature flag
    // if (!FLAGS_FC_DELETE_PROFILING_FILE)
    //     return;

    // // Skip cleanup for uninitialized paths
    // if (profiling_file_path_.empty()) {
    //     VLOG(3) << "profiling cleanup: No path specified";
    //     return;
    // }

    // std::error_code ec;

    // // Unified filesystem removal (handles both files and directories)
    // const uintmax_t removed_count = std::filesystem::remove_all(profiling_file_path_, ec);

    // // Error handling for filesystem operations
    // if (ec) {
    //     LOG(ERROR) << "profiling cleanup failed for " << profiling_file_path_ << " | Error: " << ec.message() << " ("
    //                << ec.value() << ")";
    // }
    // else {
    //     VLOG(1) << "profiling cleanup: Removed " << removed_count << " entries from " << profiling_path;
    // }
}

// get Profiling Engine instance
ProfilingEngine* ProfilingEngine::GetInstance()
{
    static ProfilingEngine instance;
    return &instance;
}

// Loads profiling database for the target device
void ProfilingEngine::LoadProfilingDB()
{
    auto db_path = GetProfilingDBPath();
    if (db_path.empty()) {
        LOG(WARNING) << "Profiling database disabled: No valid cache path available";
        db_path_.clear();  // Ensure db_path_ is properly set to empty
        return;
    }

    try {
        db_path_      = db_path;  // Store the path for later reference
        profiling_db_ = std::make_unique<ProfilingDB>(db_path);
        LOG(INFO) << "Successfully initialized profiling database at: " << db_path.string();
    }
    catch (const std::exception& e) {
        db_path_.clear();  // Clear path on failure
        FC_THROW(Fatal("Failed to initialize profiling database at {}: {}", db_path.string(), e.what()));
    }
}

// Constructs validated database path following hierarchy:
// 1. User-specified directory (FC_TUNING_DB_DIR)
// 2. Default FlashCK cache directory (FC_HOME_PATH/.flashck)
// 3. Temporary directory fallback for critical failures
std::filesystem::path ProfilingEngine::GetProfilingDBPath()
{
    // Validate environment configuration
    if (FLAGS_FC_HOME_PATH.empty()) {
        LOG(ERROR) << "FC_HOME_PATH not configured - profiling database disabled";
        return {};
    }

    // Prefer user-specified tuning database directory if available
    std::filesystem::path cache_dir;
    if (!FLAGS_FC_TUNING_DB_DIR.empty()) {
        cache_dir = std::filesystem::path(FLAGS_FC_TUNING_DB_DIR);
        LOG(INFO) << "Using custom tuning database directory: " << cache_dir.string();
    }
    else {
        cache_dir = std::filesystem::path(FLAGS_FC_HOME_PATH) / ".flashck";
        LOG(INFO) << "Using default cache directory: " << cache_dir.string();
    }

    // Ensure directory existence with proper error handling
    try {
        FileManager::CreateDirectoryIfNotExists(cache_dir);

        if (!FileManager::CheckWithRetries(cache_dir)) {
            LOG(WARNING) << "Failed to create primary database directory: " << cache_dir.string();
            return TryFallbackDBPath();
        }
    }
    catch (const std::exception& e) {
        LOG(WARNING) << "Error creating database directory " << cache_dir.string() << ": " << e.what();
        return TryFallbackDBPath();
    }

    // Construct final database path
    const std::filesystem::path db_full_path = cache_dir / "flash_ck.db";

    // Handle database flushing when requested
    if (FLAGS_FC_FLUSH_PROFILING_DB) {
        FlushExistingDB(db_full_path);
    }

    LOG(INFO) << "Profiling database path configured: " << db_full_path.string();
    return db_full_path;
}

std::filesystem::path ProfilingEngine::TryFallbackDBPath()
{
    LOG(WARNING) << "Attempting fallback database location creation";

    try {
        const std::filesystem::path temp_dir = FileManager::CreateTemporaryDirectory(".tmp_flash_ck");

        if (temp_dir.empty()) {
            LOG(FATAL) << "Critical failure: Cannot create temporary directory for database fallback";
            return {};
        }

        if (!FileManager::CheckWithRetries(temp_dir)) {
            LOG(FATAL) << "Critical failure: Fallback cache directory validation failed: " << temp_dir.string();
            return {};
        }

        const std::filesystem::path fallback_db_path = temp_dir / "flash_ck.db";
        LOG(WARNING) << "Using fallback database location: " << fallback_db_path.string();
        return fallback_db_path;
    }
    catch (const std::exception& e) {
        LOG(FATAL) << "Critical failure creating fallback database path: " << e.what();
        return {};
    }
}

// Removes existing database files with comprehensive validation
void ProfilingEngine::FlushExistingDB(const std::filesystem::path& path)
{
    if (path.empty()) {
        LOG(WARNING) << "Cannot flush database: empty path provided";
        return;
    }

    std::error_code ec;

    // Check if database file exists before attempting removal
    if (!std::filesystem::exists(path, ec)) {
        if (ec) {
            LOG(WARNING) << "Error checking database existence for " << path.string() << ": " << ec.message();
        }
        else {
            LOG(INFO) << "Database flush: file does not exist at " << path.string();
        }
        return;
    }

    // Attempt database file removal
    if (std::filesystem::remove(path, ec)) {
        LOG(INFO) << "Successfully flushed profiling database: " << path.string();
    }
    else {
        const std::string error_msg = ec ? ec.message() : "Unknown filesystem error";
        LOG(WARNING) << "Failed to flush database " << path.string() << ": " << error_msg;
    }
}

void ProfilingEngine::LoadKernelLibrary(const std::string& folder_name,
                                        const std::string& context_name,
                                        const std::string& so_file_name)
{
    // Validate input parameters
    if (folder_name.empty() || context_name.empty() || so_file_name.empty()) {
        FC_THROW(Fatal("Invalid library loading parameters: folder='{}', context='{}', file='{}'",
                       folder_name,
                       context_name,
                       so_file_name));
    }

    // Validate environment configuration
    if (FLAGS_FC_HOME_PATH.empty()) {
        FC_THROW(Fatal("FC_HOME_PATH environment variable not configured for library loading"));
    }

    // Construct full library path with proper path handling
    const std::filesystem::path lib_path =
        std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_name / so_file_name;

    LOG(INFO) << "Attempting to load kernel library: " << lib_path.string();

    // Verify library file existence with detailed error reporting
    std::error_code ec;
    if (!std::filesystem::exists(lib_path, ec)) {
        if (ec) {
            FC_THROW(Fatal("Error checking kernel library at {}: {}", lib_path.string(), ec.message()));
        }
        else {
            FC_THROW(Fatal("Kernel library not found: {}", lib_path.string()));
        }
    }

    // Verify file is readable
    if (!std::filesystem::is_regular_file(lib_path, ec)) {
        const std::string error_msg = ec ? ec.message() : "Not a regular file";
        FC_THROW(Fatal("Invalid kernel library file {}: {}", lib_path.string(), error_msg));
    }

    // Attempt dynamic loading with comprehensive error handling
    try {
        kernel_lib_ = std::make_unique<dylib>(
            lib_path.parent_path().string(), lib_path.filename().string(), dylib::NO_DECORATIONS);

        LOG(INFO) << "Successfully loaded kernel library: " << lib_path.string();
    }
    catch (const dylib::load_error& e) {
        FC_THROW(Fatal("Failed to dynamically load kernel library {}: {}", lib_path.string(), e.what()));
    }
    catch (const std::exception& e) {
        FC_THROW(Fatal("Unexpected error loading kernel library {}: {}", lib_path.string(), e.what()));
    }
}

}  // namespace flashck