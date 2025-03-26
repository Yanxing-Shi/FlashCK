#pragma once

#include <filesystem>
#include <vector>

#include "lightinfer/core/graph/node.h"
#include "lightinfer/core/profiler/base.h"
#include "lightinfer/core/utils/flags.h"

LI_DECLARE_string(LI_HOME_PATH);

namespace lightinfer {

std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>>
GenProfiler(const std::vector<Operation*>& model_ops, const DynamicProfileStrategy& strategy)
{
    std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>> results;
    for (Operation* op : model_ops) {
        if (op->has_profiler_ == true) {
            VLOG(1) << "Generate profiler for " << op->GetName();
            results.emplace_back(op->GenOpProfiler(strategy));
        }
        else {
            VLOG(1) << "Skip profiler for " << op->GetName();
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

    std::filesystem::path prefix_path = std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / context_name;

    if (!std::filesystem::exists(prefix_path)) {
        std::filesystem::create_directories(prefix_path);
    }

    for (Operation* op : model_ops) {
        if (op->has_gen_function_ == true) {
            if (exist_func.find(op->GetName()) == exist_func.end()) {
                std::filesystem::path src_path = prefix_path / (op->GetName() + ".cc");
                std::filesystem::path obj_path = prefix_path / (op->GetName() + ".o");
                file_tuples.emplace_back(src_path, obj_path);

                if (std::filesystem::exists(src_path) && std::filesystem::exists(obj_path)) {
                    LOG(INFO) << "Skip gen function for " << op->GetName();
                    continue;
                }

                // if (std::filesystem::exists(obj_path)) {
                //     LOG(WARNING) << "remove exist obj " << obj_path.string();
                //     std::filesystem::remove(obj_path);
                // }

                std::ofstream src_file(src_path.string().c_str());

                if (src_file.is_open()) {
                    src_file << op->GenOpFunction();
                    src_file.close();
                }
                else {
                    LI_THROW(Unavailable("unable to open file {}", src_path.string()));
                }
                exist_func.insert(op->GetName());
            }
        }
        else {
            VLOG(1) << "Skip gen function for " << op->GetName();
        }
    }

    LOG(INFO) << "generated " << file_tuples.size() << " function srcs";
    return file_tuples;
}

}  // namespace lightinfer