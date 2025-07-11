#pragma once

#include "flashck/core/utils/common.h"

FC_DECLARE_bool(FC_TIME_COMPILATION);

namespace flashck {

// whether to time the compilation command
inline std::string TimeCmd(const std::string& cmd)
{
    return FLAGS_FC_TIME_COMPILATION ? Sprintf("time -f 'exit_status=%x elapsed_sec=%e argv=\"%C\"' {}", cmd) : cmd;
}

// running make commands
inline void RunMakeCmds(const std::vector<std::string>& cmds,  // [0] = "make clean", [1] = "make run",
                        const std::filesystem::path&    build_dir)
{
    VLOG(1) << "Running make commands: " << cmds << " in directory: " << build_dir;

    // make clean
    subprocess::check_output(SplitStrings(cmds[0], " "));

    // make run
    subprocess::Popen popen(
        SplitStrings(cmds[1], " "), subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});
    try {
        std::string stdout_str = popen.communicate().first.buf.data();
        std::string stderr_str = popen.communicate().second.buf.data();

        VLOG(1) << "make stdout: " << stdout_str;
        VLOG(1) << "make stderr: " << stderr_str;
    }
    catch (const std::exception& e) {
        popen.kill();
        std::string stdout_str = popen.communicate().first.buf.data();
        std::string stderr_str = popen.communicate().second.buf.data();
        LOG(ERROR) << "Failed to run make commands: " << cmds[1] << "\n"
                   << "stdout: " << stdout_str << "\n"
                   << "stderr: " << stderr_str << "\n"
                   << "exception: " << e.what() << "\n";
    }
}
}  // namespace flashck