#pragma once

#include <exception>
#include <filesystem>
#include <string>
#include <tuple>

#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/printf.h"
#include "lightinfer/core/utils/string_utils.h"

#include "lightinfer/core/profiler/build_cache.h"
#include "lightinfer/core/utils/subprocess_utils.h"

LI_DECLARE_bool(LI_TIME_COMPILATION);

namespace lightinfer {

inline std::string AugmentForTrace(const std::string& cmd)
{
    // return Sprintf(
    //     'date +"{{\\"name\\": \\"$@\\", \\"ph\\": \\"B\\", \\"pid\\": \\"$$$$\\", \\"ts\\": \\"%s%6N\\"}},";' " {}; "
    //     'date +"{{\\"name\\": \\"$@\\", \\"ph\\": \\"E\\", \\"pid\\": \\"$$$$\\", \\"ts\\": \\"%s%6N\\"}},";', cmd);
    return cmd;
}

inline std::string TimeCmd(const std::string& cmd)
{
    return FLAGS_LI_TIME_COMPILATION ? Sprintf("time -f 'exit_status=%x elapsed_sec=%e argv=\"%C\"' {}", cmd) : cmd;
}

inline void RunMakeCmds(const std::vector<std::string>& cmds,
                        const int                       timeout,
                        const std::filesystem::path&    build_dir,
                        bool                            disable_cache = true)
{
    VLOG(1) << "Running make commands: " << JoinToString(cmds);
    bool        cached_results_available;
    std::string store_cache_key;

    auto build_cache_ptr = CreateBuildCache();

    if (!disable_cache) {
        std::tie(cached_results_available, store_cache_key) = build_cache_ptr->RetrieveBuildCache(cmds, build_dir);
    }
    else {
        cached_results_available = false;
        store_cache_key          = "";
    }

    if (!cached_results_available) {
        // make clean
        subprocess::check_output(SplitString(cmds[0], " "));
        // make run
        subprocess::Popen popen(
            SplitString(cmds[1], " "), subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});
        try {
            std::string stdout_str = popen.communicate().first.buf.data();
            std::string stderr_str = popen.communicate().second.buf.data();
            // ToDO::need to deal with compile error
            if (popen.retcode() == 0 && store_cache_key != "") {

                build_cache_ptr->StoreBuildCache(cmds, build_dir, store_cache_key);
            }
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
}
}  // namespace lightinfer