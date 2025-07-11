#pragma once

#include "flashck/core/utils/common.h"

FC_DECLARE_bool(FC_TIME_COMPILATION);

namespace flashck {

// whether to time the compilation command
inline std::string TimeCmd(const std::string& cmd)
{
    return FLAGS_FC_TIME_COMPILATION ? Sprintf("time -f 'exit_status=%x elapsed_sec=%e argv=\"%C\"' {}", cmd) : cmd;
}

// Enhanced make command execution with failure tracking
inline std::pair<bool, std::string>
RunMakeCmds(const std::vector<std::string>& cmds,  // [0] = "make clean", [1] = "make run"
            const std::filesystem::path&    build_dir)
{
    VLOG(1) << "Running make commands: " << cmds << " in directory: " << build_dir;

    try {
        // make clean
        subprocess::check_output(SplitStrings(cmds[0], " "));

        // make run
        subprocess::Popen popen(
            SplitStrings(cmds[1], " "), subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});

        auto        result     = popen.communicate();
        std::string stdout_str = result.first.buf.data();
        std::string stderr_str = result.second.buf.data();

        int exit_code = popen.wait();

        VLOG(1) << "make stdout: " << stdout_str;
        VLOG(1) << "make stderr: " << stderr_str;
        VLOG(1) << "make exit code: " << exit_code;

        if (exit_code == 0) {
            return {true, stdout_str};
        }
        else {
            LOG(ERROR) << "Make command failed with exit code: " << exit_code << "\nstdout: " << stdout_str
                       << "\nstderr: " << stderr_str;
            return {false, stderr_str};
        }
    }
    catch (const std::exception& e) {
        std::string error_msg = "Exception during make execution: " + std::string(e.what());
        LOG(ERROR) << error_msg;
        return {false, error_msg};
    }
}

// Parse failed compilation files from make output
inline std::vector<std::string> ParseFailedFiles(const std::string& make_output)
{
    std::vector<std::string> failed_files;
    std::istringstream       iss(make_output);
    std::string              line;

    while (std::getline(iss, line)) {
        // Look for compilation error patterns
        if (line.find("error:") != std::string::npos || line.find("fatal error:") != std::string::npos
            || (line.find("make[") != std::string::npos && line.find("Error") != std::string::npos)) {

            // Extract filename from error line
            size_t pos = line.find(".cc");
            if (pos != std::string::npos) {
                size_t start = line.rfind('/', pos);
                if (start != std::string::npos) {
                    std::string filename = line.substr(start + 1, pos - start + 2);
                    if (std::find(failed_files.begin(), failed_files.end(), filename) == failed_files.end()) {
                        failed_files.push_back(filename);
                    }
                }
            }
        }
    }

    return failed_files;
}
}  // namespace flashck