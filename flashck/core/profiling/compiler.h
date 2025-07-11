#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "flashck/core/utils/common.h"

namespace flashck {

class Compiler {
public:
    std::string GetROCmVersion();

    std::vector<std::filesystem::path> GetIncludePaths(const std::filesystem::path& dst_file_ext);

    std::vector<std::string> GetLibraryOptions(const std::string& dst_file_ext);

    std::vector<std::string> GetCompilerOptions();

    std::filesystem::path GetROCmCompilerPath();

    std::string GetCompilerCommand(const std::vector<std::string>& src_files,
                                   const std::string&              dst_file,
                                   const std::string&              dst_file_ext,
                                   const std::vector<std::string>& extra_args = {});
};

}  // namespace flashck