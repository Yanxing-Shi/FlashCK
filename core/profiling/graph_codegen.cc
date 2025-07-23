#include "core/profiling/graph_codegen.h"

#include "core/profiling/builder.h"
#include "core/profiling/gpu_profiling_runner.h"
#include "core/profiling/profiling_engine.h"

// Global configuration flags
FC_DECLARE_string(FC_HOME_PATH);      ///< FlashCK installation home directory
FC_DECLARE_bool(FC_FORCE_PROFILING);  ///< Force re-profiling flag

namespace flashck {

GenProfilerResult GraphCodeGen::CodeGenForTuning(const std::vector<Operation*>& model_ops,
                                                 const ProfilingStrategy&       strategy)
{
    VLOG(1) << "Starting tuning code generation for " << model_ops.size()
            << " operations with strategy: " << ProfilingStrategyToString(strategy);

    GenProfilerResult results;
    results.reserve(model_ops.size());

    size_t total_instances = 0;
    size_t processed_ops   = 0;

    for (auto* op : model_ops) {
        // Defensive programming: validate operation pointer
        CHECK(op != nullptr) << "Invalid null Operation pointer at index " << processed_ops;

        // Skip operations that don't require profiling
        if (!op->has_profiling_engine_) {
            VLOG(1) << "Skipping profiling for operation '" << op->GetName() << "': Profiling engine disabled";
            results.emplace_back();  // Add empty result to maintain index consistency
            continue;
        }

        VLOG(1) << "Generating profiling instances for operation '" << op->GetName() << "'";

        // Generate kernel instances for this operation
        auto op_instances = op->CodeGenForTuning(strategy);
        total_instances += op_instances.size();

        VLOG(2) << "Generated " << op_instances.size() << " instances for '" << op->GetName() << "'";
        results.emplace_back(std::move(op_instances));

        processed_ops++;
    }

    VLOG(1) << "Tuning code generation completed: " << processed_ops << " operations processed, " << total_instances
            << " total kernel instances generated";

    return results;
}

void GraphCodeGen::CodeGenAndProfiling(const std::vector<Operation*>& model_ops,
                                       const std::string&             context_name,
                                       const ProfilingStrategy&       strategy,
                                       const std::string&             folder_name,
                                       const std::string&             so_file_name)
{
    VLOG(1) << "Starting complete code generation and profiling pipeline";
    VLOG(1) << "Context: " << context_name << ", Strategy: " << ProfilingStrategyToString(strategy)
            << ", Output folder: " << folder_name;

    // Construct full library path with proper path handling
    const std::filesystem::path lib_path =
        std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_name / so_file_name;

    if (!FileManager::FileExists(lib_path) && FLAGS_FC_FORCE_PROFILING) {
        VLOG(1) << "Shared library does not exist: " << lib_path.string();
        // Phase 1: Generate profiling instances for all operations
        VLOG(1) << "Phase 1: Generating kernel instances for profiling";
        std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>> instance_files =
            CodeGenForTuning(model_ops, strategy);

        // Count total instances across all operations
        size_t total_instances = 0;
        for (const auto& op_instances : instance_files) {
            total_instances += op_instances.size();
        }
        VLOG(1) << "Generated " << total_instances << " total kernel instances across " << instance_files.size()
                << " operations";

        // Phase 2: Build profiling binaries
        VLOG(1) << "Phase 2: Building profiling binaries";
        Builder builder;
        builder.MakeTuning(instance_files, context_name);

        // Phase 3: Execute profiling and collect performance metrics
        VLOG(1) << "Phase 3: Executing profiling for performance measurement";
        auto profiling_runner = GPUProfilingRunner{Postprocesser{}};

        size_t profiled_ops = 0;
        for (Operation* op : model_ops) {
            if (op->has_profiling_engine_) {
                VLOG(2) << "Profiling operation: " << op->GetName();
                op->Tuning(profiling_runner);
                profiled_ops++;
            }
        }

        VLOG(1) << "Profiling execution completed for " << profiled_ops << " operations";

        // Wait for all profiling tasks to complete and process results
        VLOG(1) << "Processing profiling results and selecting optimal configurations";
        profiling_runner.Join();

        // Phase 4: Generate optimized runtime kernel functions
        VLOG(1) << "Phase 4: Generating optimized runtime kernel functions";
        std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples =
            CodeGenForRunning(model_ops, context_name, folder_name);

        // Phase 5: Build final shared library
        VLOG(1) << "Phase 5: Building final shared library";
        builder.MakeRunning(file_tuples, so_file_name, context_name);
    }
    else {
        VLOG(1) << "Shared library already exists: " << so_file_name
                << ", Skipping code generation and profiling as the library is already available";
    }

    // Phase 6: Load kernel library for immediate runtime availability
    VLOG(1) << "Phase 6: Loading kernel library for runtime use";
    ProfilingEngine::GetInstance()->LoadKernelLibrary(folder_name, context_name, so_file_name);

    VLOG(1) << "Complete code generation and profiling pipeline finished successfully";
}

GenFunctionResult GraphCodeGen::CodeGenForRunning(const std::vector<Operation*>& model_ops,
                                                  const std::string&             context_name,
                                                  const std::string&             folder_name)
{
    VLOG(1) << "Starting runtime code generation for " << model_ops.size() << " operations";
    VLOG(1) << "Context: " << context_name << ", Folder: " << folder_name;

    GenFunctionResult               file_tuples;
    std::unordered_set<std::string> processed_ops;

    // Configure output directory structure
    const auto prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_name;
    VLOG(2) << "Output directory: " << prefix_path.string();

    // Ensure output directory exists
    FileManager::CreateDirectoryIfNotExists(prefix_path);

    // Pre-allocate storage for efficiency
    file_tuples.reserve(model_ops.size());

    size_t generated_files = 0;
    size_t reused_files    = 0;
    size_t skipped_ops     = 0;

    for (auto* op : model_ops) {
        // Validate operation pointer
        CHECK(op != nullptr) << "Encountered null Operation pointer";

        // Skip operations that don't require runtime function generation
        if (!op->has_gen_function_) {
            VLOG(2) << "Skipping runtime codegen for '" << op->GetName() << "': Function generation disabled";
            skipped_ops++;
            continue;
        }

        const auto op_name = op->GetName();

        // Prevent duplicate processing of operations with identical names
        if (processed_ops.contains(op_name)) {
            VLOG(2) << "Skipping duplicate operation name: " << op_name;
            continue;
        }

        // Configure file paths for source and object files
        const auto src_path = prefix_path / (op_name + ".cc");
        const auto obj_path = prefix_path / (op_name + ".o");
        file_tuples.emplace_back(src_path, obj_path);

        // Check for existing valid artifacts to avoid unnecessary regeneration
        if (std::filesystem::exists(src_path) && std::filesystem::exists(obj_path)) {
            VLOG(2) << "Reusing existing artifacts for '" << op_name << "'";
            processed_ops.insert(op_name);
            reused_files++;
            continue;
        }

        // Generate optimized source code using best configuration from profiling
        VLOG(2) << "Generating optimized source code for '" << op_name << "'";
        std::string source_code = op->CodeGenForRunning();

        // Write source code to file
        FileManager::WriteFile(src_path, source_code);

        // Track processed operation
        processed_ops.insert(op_name);
        generated_files++;

        VLOG(2) << "Generated runtime function source: " << src_path.string();
    }

    // Report generation statistics
    VLOG(1) << "Runtime code generation completed:";
    VLOG(1) << "  Generated files: " << generated_files;
    VLOG(1) << "  Reused files: " << reused_files;
    VLOG(1) << "  Skipped operations: " << skipped_ops;
    VLOG(1) << "  Total output files: " << file_tuples.size();

    return file_tuples;
}

}  // namespace flashck