#pragma once

#include <filesystem>
#include <vector>

#include "flashck/core/graph/node.h"
#include "flashck/core/profiling/profiling_strategy.h"
#include "flashck/core/utils/common.h"

FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

using PathPair          = std::tuple<std::filesystem::path, std::filesystem::path>;
using OpProfilerList    = std::vector<PathPair>;
using GenProfilerResult = std::vector<OpProfilerList>;

using PathTuple         = std::tuple<std::filesystem::path, std::filesystem::path>;
using GenFunctionResult = std::vector<PathTuple>;

// Generates profiling file paths for operations supporting dynamic profiling.
GenProfilerResult GenProfiler(const std::vector<Operation*>& model_ops, const ProfilingStrategy& strategy)
{
    GenProfilerResult results;
    results.reserve(model_ops.size());

    for (auto* op : model_ops) {
        CHECK(op != nullptr) << "Invalid null Operation pointer";  // Defensive programming

        if (!op->has_profiling_engine_) {
            VLOG(1) << "Skip profiler for " << op->GetName() << ": Profiling not enabled";
            continue;
        }

        VLOG(1) << "Generate profiler for " << op->GetName()
                << " with strategy: " << ProfilingStrategyToString(strategy);
        results.emplace_back(op->GenOpProfiler(strategy));
    }

    return results;
}

// Generates source/object file pairs for operation-specific functions.
GenFunctionResult GenFunctionSource(const std::vector<Operation*>& model_ops,
                                    const std::string&             context_name,
                                    const std::string&             folder_name = "kernel_profile")
{
    GenFunctionResult               file_tuples;
    std::unordered_set<std::string> processed_ops;

    // Configure output paths
    const auto prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_name;

    try {
        std::filesystem::create_directories(prefix_path);
    }
    catch (const std::filesystem::filesystem_error& e) {
        FC_THROW(Unavailable("Failed to create directory {}: {}", prefix_path.string(), e.what()));
    }

    file_tuples.reserve(model_ops.size());

    for (auto* op : model_ops) {
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
        try {
            FileManager::WriteFile(src_path, op->GenOpFunction());
        }
        catch (const std::exception& e) {
            FC_THROW(Unavailable("Failed to write to {}: {}", src_path.string(), e.what()));
        }

        processed_ops.insert(op_name);
        LOG(INFO) << "Generated function source: " << src_path.string();
    }

    LOG(INFO) << "Total generated source files: " << file_tuples.size();
    return file_tuples;
}

}  // namespace flashck