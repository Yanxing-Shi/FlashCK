#pragma once

#include <sstream>

#include "core/profiling/builder.h"
#include "core/profiling/compiler.h"
#include "core/profiling/graph_codegen.h"
#include "core/profiling/profiling_db.h"

namespace flashck {

/**
 * @class ProfilingEngine
 * @brief Singleton orchestrator for kernel profiling and optimization pipeline
 *
 * The ProfilingEngine provides a centralized interface for managing the complete
 * kernel optimization workflow from initial code generation through profiling
 * to optimized runtime deployment. It coordinates multiple subsystems including:
 *
 * - **Database Management**: Caches optimal kernel configurations for reuse
 * - **Code Generation**: Orchestrates graph-level kernel generation and compilation
 * - **Library Management**: Handles dynamic loading of optimized kernel libraries
 * - **Resource Coordination**: Manages shared resources across profiling components
 *
 * Key Features:
 * - Thread-safe singleton pattern for global access
 * - Lazy initialization with proper error handling
 * - Database fallback mechanisms for robust operation
 * - Integrated library loading for seamless runtime deployment
 * - Comprehensive logging and error reporting
 *
 * Usage Pattern:
 * ```cpp
 * auto* engine = ProfilingEngine::GetInstance();
 * engine->LoadProfilingDB();
 * auto* db = engine->GetProfilingDB();
 * auto* codegen = engine->GetGraphCodeGen();
 * ```
 */
class ProfilingEngine {
public:
    /**
     * @brief Initialize profiling engine with database loading
     *
     * Constructs the profiling engine singleton and initializes the profiling
     * database. Throws on critical initialization failures that would prevent
     * proper operation of the profiling system.
     *
     * @throws Fatal if database initialization fails
     */
    explicit ProfilingEngine();

    // Disable copy and move operations to maintain singleton pattern
    ProfilingEngine(const ProfilingEngine&)            = delete;
    ProfilingEngine& operator=(const ProfilingEngine&) = delete;
    ProfilingEngine(ProfilingEngine&&)                 = delete;
    ProfilingEngine& operator=(ProfilingEngine&&)      = delete;

    /**
     * @brief Clean up profiling engine resources
     *
     * Destructor handles cleanup of managed resources including database
     * connections and loaded kernel libraries. Commented-out cleanup code
     * shows optional file system cleanup that can be enabled if needed.
     */
    ~ProfilingEngine();

    /**
     * @brief Get the singleton ProfilingEngine instance
     * @return Pointer to the singleton ProfilingEngine instance
     *
     * Provides thread-safe access to the singleton ProfilingEngine instance
     * using static local variable initialization. The instance is created
     * on first access and persists for the program lifetime.
     *
     * @note This method is thread-safe in C++11 and later
     */
    static ProfilingEngine* GetInstance();

    /**
     * @brief Construct validated database path with fallback mechanisms
     * @return Filesystem path to the profiling database file
     *
     * Determines the optimal location for the profiling database following
     * a hierarchy of preferences:
     * 1. User-specified directory (FC_TUNING_DB_DIR)
     * 2. Default FlashCK cache directory (FC_HOME_PATH/.flashck)
     * 3. Temporary directory fallback for critical failures
     *
     * Includes automatic directory creation and validation to ensure the
     * database can be successfully created and accessed.
     */
    std::filesystem::path GetProfilingDBPath();

    /**
     * @brief Initialize and load the profiling database
     *
     * Creates the ProfilingDB instance using the validated database path.
     * Handles database initialization errors gracefully and provides
     * comprehensive error reporting for debugging failed initialization.
     *
     * @throws Fatal if database creation fails after path validation
     */
    void LoadProfilingDB();

    /**
     * @brief Create fallback database path in temporary directory
     * @return Filesystem path to fallback database location
     *
     * Used when the primary database location cannot be created or accessed.
     * Creates a temporary directory with appropriate permissions and returns
     * a database path within that directory. Critical for system robustness
     * in environments with restricted filesystem access.
     *
     * @throws Fatal if fallback directory creation fails
     */
    std::filesystem::path TryFallbackDBPath();

    /**
     * @brief Remove existing database file for fresh initialization
     * @param path Filesystem path to the database file to remove
     *
     * Safely removes an existing database file when fresh initialization
     * is requested (FC_FLUSH_PROFILING_DB flag). Includes proper error
     * handling and logging for failed removal operations.
     */
    void FlushExistingDB(const std::filesystem::path& path);

    /**
     * @brief Load optimized kernel library for runtime execution
     * @param folder_name Root directory name containing kernel libraries
     * @param context_name Context-specific subdirectory for organization
     * @param so_file_name Shared library filename to load
     *
     * Dynamically loads a compiled kernel library (.so file) for runtime
     * execution. Validates the library path, verifies file existence, and
     * handles dynamic loading errors gracefully with detailed error reporting.
     *
     * The library is loaded using the dylib wrapper for cross-platform
     * compatibility and stored in the engine for subsequent kernel execution.
     *
     * @throws Fatal if library path is invalid, file doesn't exist, or loading fails
     */
    void
    LoadKernelLibrary(const std::string& folder_name, const std::string& context_name, const std::string& so_file_name);

    /**
     * @brief Get profiling database instance for cache operations
     * @return Pointer to ProfilingDB instance, or nullptr if not initialized
     *
     * Provides access to the profiling database for querying cached optimal
     * configurations and inserting newly discovered optimal kernels. The
     * database enables significant performance improvements by avoiding
     * redundant profiling of previously optimized configurations.
     */
    ProfilingDB* GetProfilingDB()
    {
        return profiling_db_.get();
    }

    /**
     * @brief Get loaded kernel library for function execution
     * @return Pointer to dylib instance, or nullptr if no library loaded
     *
     * Provides access to the dynamically loaded kernel library for executing
     * optimized GPU kernels at runtime. The library contains compiled kernel
     * functions that have been optimized through the profiling pipeline.
     */
    dylib* GetKernelLibrary()
    {
        return kernel_lib_.get();
    }

    /**
     * @brief Get graph code generator for kernel compilation
     * @return Pointer to GraphCodeGen instance
     *
     * Provides access to the graph-level code generation orchestrator for
     * transforming computational graphs into optimized GPU kernels. The
     * code generator handles the complete pipeline from graph analysis
     * through kernel compilation and optimization.
     */
    GraphCodeGen* GetGraphCodeGen()
    {
        return graph_codegen_.get();
    }

    /**
     * @brief Check if the profiling engine is fully initialized
     * @return True if all critical components are initialized, false otherwise
     *
     * Validates that the profiling engine has successfully initialized all
     * its core components including the graph code generator and profiling
     * database. Use this method to verify engine readiness before attempting
     * optimization operations.
     */
    bool IsInitialized() const
    {
        return (graph_codegen_ != nullptr) && (profiling_db_ != nullptr || db_path_.empty());
    }

    /**
     * @brief Get detailed initialization status for debugging
     * @return String describing the current initialization state
     *
     * Provides comprehensive information about which components have been
     * successfully initialized and which may have failed. Useful for
     * debugging initialization issues and verifying system configuration.
     */
    std::string GetInitializationStatus() const
    {
        std::ostringstream status;
        status << "ProfilingEngine Status:\n";
        status << "  Graph CodeGen: " << (graph_codegen_ ? "Initialized" : "Not Initialized") << "\n";
        status << "  Profiling DB: " << (profiling_db_ ? "Initialized" : "Not Initialized") << "\n";
        status << "  DB Path: " << (db_path_.empty() ? "Not Set" : db_path_.string()) << "\n";
        status << "  Kernel Library: " << (kernel_lib_ ? "Loaded" : "Not Loaded") << "\n";
        status << "  Overall Status: " << (IsInitialized() ? "Ready" : "Not Ready");
        return status.str();
    }

    /**
     * @brief Unload current kernel library
     *
     * Safely unloads the currently loaded kernel library, freeing associated
     * resources. This is useful when switching between different kernel
     * libraries or when explicit cleanup is required before loading a new
     * library version.
     */
    void UnloadKernelLibrary()
    {
        if (kernel_lib_) {
            kernel_lib_.reset();
            VLOG(1) << "Kernel library unloaded successfully";
        }
    }

private:
    /// Database file path for profiling result caching
    std::filesystem::path db_path_;

    /// Profiling database for optimal configuration caching
    std::unique_ptr<ProfilingDB> profiling_db_;

    /// Graph-level code generation orchestrator
    std::unique_ptr<GraphCodeGen> graph_codegen_;

    /// Dynamically loaded kernel library for runtime execution
    std::unique_ptr<dylib> kernel_lib_;
};

}  // namespace flashck