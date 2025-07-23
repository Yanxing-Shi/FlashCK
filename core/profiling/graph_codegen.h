#pragma once

#include <filesystem>
#include <vector>

#include "core/graph/node.h"
#include "core/profiling/profiling_helper.h"
#include "core/utils/common.h"

namespace flashck {

// ============================================================================
// Type Aliases for Code Generation Results
// ============================================================================

/// Pair of source file path and object file path for a single kernel instance
using PathPair = std::tuple<std::filesystem::path, std::filesystem::path>;

/// List of kernel instances (path pairs) for a single operation
using OpProfilerList = std::vector<PathPair>;

/// Results from tuning code generation: list of operations, each with multiple instances
using GenProfilerResult = std::vector<OpProfilerList>;

/// Pair of source file path and object file path for runtime kernel functions
using PathTuple = std::tuple<std::filesystem::path, std::filesystem::path>;

/// Results from runtime code generation: list of operation function files
using GenFunctionResult = std::vector<PathTuple>;

/**
 * @class GraphCodeGen
 * @brief High-level orchestrator for graph-based kernel code generation
 *
 * The GraphCodeGen class provides a unified interface for transforming
 * computational graph operations into optimized GPU kernels. It handles
 * the complete workflow from initial code generation through profiling-based
 * optimization to final shared library creation for runtime deployment.
 *
 * Key Responsibilities:
 * - Generate multiple kernel instances for profiling and tuning
 * - Coordinate compilation and profiling execution
 * - Select optimal kernel configurations based on performance metrics
 * - Generate final runtime kernel functions with optimal parameters
 * - Create shared libraries for runtime kernel loading
 *
 * Workflow Stages:
 * 1. Tuning: Generate multiple kernel variants for performance evaluation
 * 2. Profiling: Compile and execute kernels to measure performance
 * 3. Selection: Choose optimal configurations based on metrics
 * 4. Runtime: Generate final optimized kernel functions
 * 5. Packaging: Create shared libraries for deployment
 */
class GraphCodeGen {
public:
    /**
     * @brief Generate kernel instances for profiling and tuning
     * @param model_ops Vector of operations to generate kernels for
     * @param strategy Profiling strategy controlling instance generation scope
     * @return Results containing generated source/object file paths for each operation
     *
     * Generates multiple kernel instances per operation for comprehensive
     * performance evaluation. Each operation may produce multiple kernel
     * variants with different optimization parameters, tile sizes, or
     * algorithmic approaches for profiling comparison.
     *
     * The profiling strategy controls the breadth of instance generation:
     * - kMax: Generate comprehensive set of instances for thorough evaluation
     * - kDefault: Generate standard set of commonly effective instances
     * - kFast: Generate minimal set for quick evaluation
     *
     * Generated files are organized by operation and placed in structured
     * directories for subsequent compilation and profiling stages.
     */
    GenProfilerResult CodeGenForTuning(const std::vector<Operation*>& model_ops,
                                       const ProfilingStrategy&       strategy = ProfilingStrategy::kMax);

    /**
     * @brief Complete end-to-end code generation, profiling, and optimization pipeline
     * @param model_ops Vector of operations to process
     * @param context_name Unique context identifier for file organization
     * @param strategy Profiling strategy for instance generation
     * @param folder_name Root directory name for generated artifacts
     *
     * Executes the complete kernel optimization pipeline:
     *
     * 1. **Instance Generation**: Creates multiple kernel variants per operation
     * 2. **Compilation**: Builds all kernel instances into profiling binaries
     * 3. **Profiling Execution**: Runs kernels and collects performance metrics
     * 4. **Optimization Selection**: Analyzes results and selects best configurations
     * 5. **Runtime Generation**: Creates optimized kernel functions for deployment
     * 6. **Library Creation**: Compiles final kernels into shared library
     * 7. **Dynamic Loading**: Loads library for immediate runtime availability
     *
     * This method provides a complete turnkey solution for kernel optimization,
     * handling all intermediate steps and coordination between subsystems.
     *
     * @note This operation can be time-intensive for large graphs or comprehensive
     *       profiling strategies due to compilation and execution overhead.
     */
    void CodeGenAndProfiling(const std::vector<Operation*>& model_ops,
                             const std::string&             context_name,
                             const ProfilingStrategy&       strategy     = ProfilingStrategy::kMax,
                             const std::string&             folder_name  = "kernel_profile",
                             const std::string&             so_file_name = "generated_kernel.so");

    /**
     * @brief Generate optimized runtime kernel functions from profiled results
     * @param model_ops Vector of operations to generate runtime functions for
     * @param context_name Context identifier for file organization
     * @param folder_name Directory name for generated source files
     * @return Vector of source/object file path pairs for runtime compilation
     *
     * Generates final optimized kernel source code using the best configurations
     * determined during the profiling phase. Each operation produces a single
     * optimized kernel function incorporating the optimal parameters discovered
     * through performance evaluation.
     *
     * Key Features:
     * - **Optimal Parameter Integration**: Uses best tile sizes, algorithms, etc.
     * - **Duplicate Prevention**: Avoids regenerating identical operation kernels
     * - **Artifact Reuse**: Skips regeneration if valid source/object files exist
     * - **Structured Organization**: Places files in hierarchical directory structure
     * - **Performance Logging**: Reports generation progress and file counts
     *
     * Generated source files are ready for compilation into the final shared
     * library that will be used for actual kernel execution in production.
     *
     * @note This method should typically be called after profiling has completed
     *       and optimal configurations have been determined and cached.
     */
    GenFunctionResult CodeGenForRunning(const std::vector<Operation*>& model_ops,
                                        const std::string&             context_name,
                                        const std::string&             folder_name = "kernel_profile");
};

}  // namespace flashck