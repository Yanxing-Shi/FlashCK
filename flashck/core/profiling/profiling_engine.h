#pragma once

#include "flashck/core/profiling/builder.h"
#include "flashck/core/profiling/compiler.h"
#include "flashck/core/profiling/graph_codegen.h"
#include "flashck/core/profiling/profiling_db.h"

namespace flashck {

class ProfilingEngine {
public:
    explicit ProfilingEngine();
    ProfilingEngine(const ProfilingEngine&)            = delete;
    ProfilingEngine& operator=(const ProfilingEngine&) = delete;
    ProfilingEngine(ProfilingEngine&&)                 = delete;
    ProfilingEngine& operator=(ProfilingEngine&&)      = delete;

    ~ProfilingEngine();

    static ProfilingEngine* GetInstance();

    // Generate kernel for the given operation kind and problem
    void GenerateKernel(const CodeGenKind& op_kind, const std::variant<NormProblem>& problem);

    // Constructs validated cache path following hierarchy:
    std::filesystem::path GetProfilingDBPath();

    // Initializes the profile cache database
    void LoadProfilingDB();

    std::filesystem::path TryFallbackDBPath();

    void FlushExistingDB(const std::filesystem::path& path);

    void CodegenAndProfileKernel(const std::vector<Operation*>& model_ops,
                                 const std::string&             context_name,
                                 const ProfilingStrategy&       strategy);

    // load kernel library
    void
    LoadKernelLibrary(const std::string& folder_name, const std::string& context_name, const std::string& so_file_name);

    // get profiling performance database
    ProfilingDB* GetProfilingDB()
    {
        return profiling_db_.get();
    }

    // get profiling engine compiler
    Compiler* GetCompiler()
    {
        return compiler_.get();
    }

private:
    std::filesystem::path db_path_;

    std::unique_ptr<Compiler>    compiler_;
    std::unique_ptr<ProfilingDB> profiling_db_;

    std::map<NormKind, std::map<std::string, NormCodeGen>> norm_instance_map_;
    std::unique_ptr<dylib>                                 kernel_lib_;
};

}  // namespace flashck