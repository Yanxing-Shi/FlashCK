#include "flashck/core/profiling/graph_codegen.h"

#include "flashck/core/profiling/builder.h"
#include "flashck/core/profiling/profiling_engine.h"

FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

GenProfilerResult GraphCodeGen::CodeGenForTuning(const std::vector<Operation*>& model_ops,
                                                 const ProfilingStrategy&       strategy)
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
        results.emplace_back(op->CodeGenForTuning(strategy));
    }

    return results;
}

void GraphCodeGen::CodeGenAndProfiling(const std::vector<Operation*>& model_ops,
                                       const std::string&             context_name,
                                       const ProfilingStrategy&       strategy,
                                       const std::string&             folder_name)
{
    // step1: profiling instance file generation
    std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>> instance_files =
        CodeGenForTuning(model_ops, strategy);

    // step.2 builder instance
    // generate makefile and other necessary files
    VLOG(1) << "Profiler generated " << instance_files.size() << " model operations";
    Builder builder;
    builder.MakeTuning(instance_files, context_name);

    // step.3 run profiling
    auto profiling_runner = GPUProfilingRunner{Postprocesser{}};
    for (Operation* op : model_ops) {
        op->Tuning(profiling_runner);
    }
    profiling_runner.Join();

    // step.3 gen kernel source function
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples =
        CodeGenForRunning(model_ops, context_name);

    // step.4 build kernel source function
    builder.MakeRunning(file_tuples, "generated_kernel.so", context_name);

    // step.5 read dll and load functio
    ProfilingEngine::GetInstance()->LoadKernelLibrary("kernel_profile", context_name, "generated_kernel.so");
}

// Generates source/object file pairs for operation-specific functions.
GenFunctionResult GraphCodeGen::CodeGenForRunning(const std::vector<Operation*>& model_ops,
                                                  const std::string&             context_name,
                                                  const std::string&             folder_name)
{
    GenFunctionResult               file_tuples;
    std::unordered_set<std::string> processed_ops;

    // Configure output paths
    const auto prefix_path = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_name;
    FileManager::CreateDirectoryIfNotExists(prefix_path);

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

        FileManager::WriteFile(src_path, op->CodeGenForRunning());

        processed_ops.insert(op_name);
        LOG(INFO) << "Generated function source: " << src_path.string();
    }

    LOG(INFO) << "Total generated source files: " << file_tuples.size();
    return file_tuples;
}

}  // namespace flashck