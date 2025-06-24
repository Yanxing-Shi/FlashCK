#include "flashck/core/utils/enforce.h"

#include <dlfcn.h>
#include <execinfo.h>

#include <glog/logging.h>

#include "flashck/core/utils/flags.h"

LI_DECLARE_int32(call_stack_level);

namespace flashck {

// FLAGS_call_stack_level>1 means showing c++ call stack
int GetCallStackLevel()
{
    return FLAGS_call_stack_level;
}

// internal throw warning
void InternalThrowWarning(const std::string& msg)
{
    LOG(WARNING) << "WARNING :" << msg;
}

std::string GetCurrentTraceBackString(bool for_signal)
{
    std::ostringstream sout;

    if (!for_signal) {
        sout << "\n\n--------------------------------------\n";
        sout << "C++ Traceback (most recent call last):";
        sout << "\n--------------------------------------\n";
    }

    static constexpr int TRACE_STACK_LIMIT = 100;

    std::array<void*, TRACE_STACK_LIMIT> call_stack;
    auto                                 size    = backtrace(call_stack.data(), TRACE_STACK_LIMIT);
    auto                                 symbols = backtrace_symbols(call_stack.data(), size);
    Dl_info                              info;
    int                                  idx = 0;
    // `for_signal` used to remove the stack trace introduced by
    // obtaining the error stack trace when the signal error occurred,
    // that is not related to the signal error self, remove it to
    // avoid misleading users and developers
    int end_idx = for_signal ? 2 : 0;
    for (int i = size - 1; i >= end_idx; --i) {
        if (dladdr(call_stack[i], &info) && info.dli_sname) {
            auto        demangled = info.dli_sname;
            std::string path(info.dli_fname);
            // C++ traceback info are from core.so
            if (path.substr(path.length() - 3).compare(".so") == 0) {
                sout << fmt::format("{} {}\n", idx++, demangled);
            }
        }
    }
    free(symbols);
    return sout.str();
}

// Simplify error type format
std::string SimplifyErrorTypeFormat(const std::string& str)
{
    std::ostringstream sout;
    size_t             type_end_pos = str.find(':', 0);
    if (type_end_pos == std::string::npos) {
        sout << str;
    }
    else {
        // Remove "Error:", add "()""
        sout << "(" << str.substr(0, type_end_pos - 5) << ")" << str.substr(type_end_pos + 1);
    }
    return sout.str();
}

}  // namespace flashck
