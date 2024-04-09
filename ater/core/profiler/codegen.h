#pragma once

#include <filesystem>
#include <vector>

#include "ater/core/graph/node.h"
#include "ater/core/module/operations/gemm_universal/gemm_common_op.h"
#include "ater/core/profiler/base.h"
#include "ater/core/utils/flags.h"

ATER_DECLARE_string(ATER_HOME_PATH);

namespace ater {

std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>
GenProfiler(const std::vector<Operation*>& model_ops,
            const DynamicProfileStrategy&  strategy = DynamicProfileStrategy::MAX)
{
    std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>> results;
    for (Operation* op : model_ops) {
        if (op->has_profiler_ == true) {
            VLOG(1) << "Generate profiler for " << op->GetName();
            results.emplace_back(op->GenOpProfiler(strategy));
        }
    }

    return results;
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GenFunctionSource(const std::vector<Operation*>& model_ops,
                  const std::string&             context_name,
                  const std::string&             folder_name = "kernel_profile")
{
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples;
    std::unordered_set<std::string>                                       exist_func;

    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_ATER_HOME_PATH) / folder_name / context_name;
    if (!std::filesystem::exists(prefix_path)) {
        std::filesystem::create_directories(prefix_path);
    }

    for (Operation* op : model_ops) {
        if (exist_func.find(op->GetName()) == exist_func.end()) {
            std::filesystem::path src_path = prefix_path / (op->GetName() + ".cc");
            std::filesystem::path obj_path = prefix_path / (op->GetName());
            file_tuples.emplace_back(src_path, obj_path);

            std::ofstream src_file(src_path.string().c_str());
            if (src_file.is_open()) {
                src_file << op->GenOpFunction();
                src_file.close();
            }
            else {
                ATER_THROW(Unavailable("unable to open file {}", src_path.string()));
            }
            exist_func.insert(op->GetName());
        }
    }

    LOG(INFO) << "generated " << file_tuples.size() << " function srcs";
    return file_tuples;
}

}  // namespace ater