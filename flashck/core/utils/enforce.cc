#include "flashck/core/utils/enforce.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <sstream>

#include <glog/logging.h>

#include "flashck/core/utils/flags.h"

FC_DECLARE_int32(FC_CALL_STACK_LEVEL);

namespace flashck {

// Get the current call stack level setting
int GetCallStackLevel()
{
    return FLAGS_FC_CALL_STACK_LEVEL;
}

// Log warning message to the logging system
void InternalThrowWarning(const std::string& msg)
{
    LOG(WARNING) << "WARNING: " << msg;
}

// Get a detailed stack trace as a formatted string
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
    int                                  size = backtrace(call_stack.data(), TRACE_STACK_LIMIT);

    if (size <= 0) {
        sout << "(No stack trace available)\n";
        return sout.str();
    }

    char** symbols = backtrace_symbols(call_stack.data(), size);
    if (!symbols) {
        sout << "(Failed to get symbol information)\n";
        return sout.str();
    }

    Dl_info info;
    int     idx = 0;

    // Skip stack frames introduced by signal handling if for_signal is true
    int end_idx = for_signal ? 2 : 0;

    for (int i = size - 1; i >= end_idx; --i) {
        if (dladdr(call_stack[i], &info) && info.dli_sname) {
            std::string path(info.dli_fname ? info.dli_fname : "");

            // Only show stack frames from shared libraries (.so files)
            if (path.length() >= 3 && path.substr(path.length() - 3) == ".so") {
                sout << "[" << idx++ << "] " << info.dli_sname << " (" << path << ")\n";
            }
        }
        else {
            // Fallback to raw symbol if dladdr fails
            sout << "[" << idx++ << "] " << (symbols[i] ? symbols[i] : "<unknown>") << "\n";
        }
    }

    free(symbols);
    return sout.str();
}

// Simplify error message format for user-friendly display
std::string SimplifyErrorTypeFormat(const std::string& str)
{
    if (str.empty()) {
        return str;
    }

    std::ostringstream sout;
    size_t             type_end_pos = str.find(':', 0);

    if (type_end_pos == std::string::npos) {
        // No colon found, return as is
        sout << str;
    }
    else {
        // Convert "ErrorType: message" to "(ErrorType) message"
        std::string error_type = str.substr(0, type_end_pos);

        // Remove "Error" suffix if present
        if (error_type.length() > 5 && error_type.substr(error_type.length() - 5) == "Error") {
            error_type = error_type.substr(0, error_type.length() - 5);
        }

        sout << "(" << error_type << ")" << str.substr(type_end_pos + 1);
    }

    return sout.str();
}

}  // namespace flashck
