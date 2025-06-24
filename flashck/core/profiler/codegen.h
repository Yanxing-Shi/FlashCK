#pragma once

#include <filesystem>
#include <vector>

#include "flashck/core/graph/node.h"
#include "flashck/core/profiler/base.h"
#include "flashck/core/utils/flags.h"

LI_DECLARE_string(LI_HOME_PATH);

namespace flashck {

// Type aliases to simplify complex nested types
using PathPair          = std::tuple<std::filesystem::path, std::filesystem::path>;
using OpProfilerList    = std::vector<PathPair>;
using GenProfilerResult = std::vector<OpProfilerList>;

using PathTuple         = std::tuple<std::filesystem::path, std::filesystem::path>;
using GenFunctionResult = std::vector<PathTuple>;

/// Generates profiling file paths for operations supporting dynamic profiling.
///
/// Iterates through model operations, generating path pairs (input/output)
/// for operations with profiling enabled. Skips others with debug logging.
///
/// @param model_ops List of operations in the model
/// @param strategy Profiling configuration strategy
/// @return Nested vector of file path pairs per profiled operation
GenProfilerResult GenProfiler(const std::vector<Operation*>& model_ops, const DynamicProfileStrategy& strategy)
{
    GenProfilerResult results;
    results.reserve(model_ops.size());  // Pre-allocate to minimize reallocations

    for (const auto* op : model_ops) {
        CHECK(op != nullptr) << "Invalid null Operation pointer";  // Defensive programming

        if (!op->has_profiler_) {
            VLOG(1) << "Skip profiler for " << op->GetName() << ": Profiling not enabled";
            continue;
        }

        VLOG(1) << "Generate profiler for " << op->GetName() << " with strategy: " << strategy.ToString();
        results.emplace_back(op->GenOpProfiler(strategy));
    }

    return results;
}

/// Generates source/object file pairs for operation-specific functions.
///
/// Creates output directory structure and handles existing files validation.
/// Skips operations without code generation capability or duplicate names.
///
/// @param model_ops List of model operations to process
/// @param context_name Namespace for output files
/// @param folder_name Output directory name (default: "kernel_profile")
/// @return Vector of (source_path, object_path) tuples
/// @throws Unavailable If file creation fails
GenFunctionResult GenFunctionSource(const std::vector<Operation*>& model_ops,
                                    const std::string&             context_name,
                                    const std::string&             folder_name = "kernel_profile")
{
    GenFunctionResult               file_tuples;
    std::unordered_set<std::string> processed_ops;

    // Configure output paths
    const auto prefix_path = std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / context_name;

    try {
        std::filesystem::create_directories(prefix_path);
    }
    catch (const std::filesystem::filesystem_error& e) {
        LI_THROW(Unavailable("Failed to create directory {}: {}", prefix_path.string(), e.what()));
    }

    file_tuples.reserve(model_ops.size());  // Prevent multiple reallocations

    for (const auto* op : model_ops) {
        CHECK(op != nullptr) << "Received null Operation pointer";

        if (!op->has_gen_function_) {
            VLOG(1) << "Skipping codegen for " << op->GetName() << ": Function generation disabled";
            continue;
        }

        const auto op_name = op->GetName();
        if (processed_ops.contains(op_name)) {
            VLOG(2) << "Duplicate operation name skipped: " << op_name;
            continue;
        }

        // Configure file paths
        const auto src_path = prefix_path / (op_name + ".cc");
        const auto obj_path = prefix_path / (op_name + ".o");
        file_tuples.emplace_back(src_path, obj_path);

        // Check for existing artifacts
        if (std::filesystem::exists(src_path) && std::filesystem::exists(obj_path)) {
            LOG(INFO) << "Reusing existing artifacts for " << op_name;
            processed_ops.insert(op_name);
            continue;
        }

        // Generate source code
        std::ofstream src_file(src_path);
        if (!src_file) {
            LI_THROW(Unavailable("Failed to open source file: {}", src_path.string()));
        }

        try {
            src_file << op->GenOpFunction();
        }
        catch (const std::exception& e) {
            LI_THROW(Unavailable("Failed to write to {}: {}", src_path.string(), e.what()));
        }

        processed_ops.insert(op_name);
        LOG(INFO) << "Generated function source: " << src_path.string();
    }

    LOG(INFO) << "Total generated source files: " << file_tuples.size();
    return file_tuples;
}

}  // namespace flashck