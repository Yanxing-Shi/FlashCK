#include "flashck/core/profiling/profiling_engine.h"

#include "flashck/core/utils/common.h"

FC_DECLARE_string(FC_HOME_PATH);
FC_DECLARE_string(FC_PROFILING_DB_DIR);
FC_DECLARE_bool(FC_FLUSH_PROFILING_DB);
FC_DECLARE_bool(FC_DELETE_PROFILING_FILE);

namespace flashck {

ProfilingEngine::ProfilingEngine()
{
    LOG(INFO) << "Initializing Profiling Engine on device: " << GetDeviceName();
    try {
        LoadProfilingDB();
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "profiling cache initialization failed: " << e.what();
        throw;
    }
}

ProfilingEngine::~ProfilingEngine()
{
    // Early exit if profiling cleanup is disabled via feature flag
    if (!FLAGS_FC_DELETE_PROFILING_FILE)
        return;

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

// Generates kernel operations and populates instance maps
void ProfilingEngine::GenerateKernel(const CodeGenKind& code_gen_kind, const std::variant<NormProblem>& problem)
{
    switch (code_gen_kind) {
        case CodeGenKind::Norm:
            norm_instance_map_ = NormEmitter::GetInstance()->GenerateInstances(std::get<NormProblem>(problem));
            break;
        default:
            break;
    }
}

// Loads profiling database for the target device
void ProfilingEngine::LoadProfilingDB()
{
    auto db_path = GetProfilingDBPath();
    if (db_path.empty()) {
        LOG(WARNING) << "profiling database disabled: No valid cache path available";
        return;
    }

    try {
        profiling_db_ = std::make_unique<ProfilingDB>(db_path);
        LOG(INFO) << "Initialized profiling database at: " << db_path.string();
    }
    catch (const std::exception& e) {
        FC_THROW(Fatal("Failed to initialize profiling database: {}", e.what()));
    }
}

// Constructs validated cache path following hierarchy:
std::filesystem::path ProfilingEngine::GetProfilingDBPath()
{
    // Configure base paths
    if (FLAGS_FC_PROFILING_DB_DIR.empty()) {
        LOG(WARNING) << "profiling cache directory not set";
        return TryFallbackDBPath();
    }
    const std::filesystem::path default_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / ".flashck";
    if (default_path.empty()) {
        LOG(ERROR) << "No valid base path available for profiling database";
        return {};
    }

    // Ensure directory existence
    if (!CheckWithRetries(default_path)) {
        LOG(WARNING) << "Failed to create primary database directory: " << default_path.string();
        return TryFallbackDBPath();
    }

    // Construct final database path
    const std::filesystem::path db_full_path = default_path / "flash_ck.db";

    // Handle database flushing
    if (FLAGS_FC_FLUSH_PROFILING_DB) {
        FlushExistingDB(db_full_path);
    }

    return db_full_path;
}

std::filesystem::path ProfilingEngine::TryFallbackDBPath()
{
    const std::filesystem::path temp_dir = CreateTemporaryDirectory(".tmp_flash_ck");

    if (temp_dir.empty() || !CheckWithRetries(temp_dir)) {
        LOG(FATAL) << "Critical failure: Cannot create fallback cache directory";
        return {};
    }

    LOG(WARNING) << "Using fallback cache location: " << temp_dir.string();
    return temp_dir / ".flash_ck";
}

// Removes existing cache files with validation
void ProfilingEngine::FlushExistingDB(const std::filesystem::path& path)
{
    std::error_code ec;
    if (std::filesystem::remove(path, ec)) {
        LOG(INFO) << "Successfully flushed cache: " << path.string();
        return;
    }

    LOG(WARNING) << "Cache flush failed for " << path.string() << ": " << (ec ? ec.message() : "Unknown error");
}

void ProfilingEngine::CodegenAndProfileKernel(const std::vector<Operation*>& model_ops,
                                              const std::string&             context_name,
                                              const ProfilingStrategy&       strategy)
{
    // step1: gen profiler
    std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>> graph_generated_profilers =
        GenProfiler(model_ops, strategy);

    // // step.2 profile result
    VLOG(1) << "Profiler generated " << graph_generated_profilers.size() << " model operations";
    Builder builder;
    builder.MakeTuning(graph_generated_profilers, context_name);
    auto profiling_runner = GPUProfilingRunner{Postprocesser{}};
    for (Operation* op_ptr : model_ops) {
        op_ptr->Profile(profiling_runner);
    }
    profiling_runner.Join();

    // step3 gen kernel source function
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples =
        GenFunctionSource(model_ops, context_name);
    builder.MakeRunning(file_tuples, "generated_kernel.so", context_name);

    // read dll and load function
    ProfilingEngine::GetInstance()->LoadKernelLibrary("kernel_profile", context_name, "generated_kernel.so");
}

void ProfilingEngine::LoadKernelLibrary(const std::string& folder_name,
                                        const std::string& context_name,
                                        const std::string& so_file_name)
{
    // Validate environment configuration
    if (FLAGS_FC_HOME_PATH.empty()) {
        FC_THROW(Fatal("FC_HOME environment path not configured"));
    }

    // Construct full library path
    const std::filesystem::path lib_path =
        std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_name / so_file_name;

    // Verify library existence
    if (!std::filesystem::exists(lib_path)) {
        FC_THROW(Fatal("Kernel library not found at {}", lib_path.string()));
    }

    // Attempt dynamic loading
    try {
        kernel_lib_ = std::make_unique<dylib>(
            lib_path.parent_path().string(), lib_path.filename().string(), dylib::no_filename_decorations);
    }
    catch (const dylib::load_error& e) {
        FC_THROW(Fatal("Failed to load {}: {}", lib_path.string(), e.what()));
    }
}

}  // namespace flashck