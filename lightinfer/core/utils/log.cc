#include "lightinfer/core/utils/log.h"

#include <mutex>

static std::once_flag g_glog_init_flag;

namespace lightinfer {

// void PrefixFormatter(std::ostream& s, const google::LogMessage& m, void* data)
// {
//     // clang-format off
//     s << "["
//         << "[LI]" << "("
//         << google::GetLogSeverityName(m.severity())[0]
//         << ")" << ' '
//         << std::setw(4) << 1900 + m.time().year()
//         << std::setw(2) << 1 + m.time().month()
//         << std::setw(2) << m.time().day()
//         << ' '
//         << std::setw(2) << m.time().hour() << ':'
//         << std::setw(2) << m.time().min()  << ':'
//         << std::setw(2) << m.time().sec() << "."
//         << std::setw(6) << m.time().usec()
//         << ' '
//         << std::setfill(' ') << std::setw(5)
//         << m.thread_id() << std::setfill('0')
//         << ' '
//         << m.basename() << ':' << m.line() << "]";

//     // clang-format on
// }

void InitGLOG(char** argv)
{
    std::call_once(g_glog_init_flag, [&]() {
        FLAGS_logtostderr = true;
        // google::InstallPrefixFormatter(&PrefixFormatter);
        google::InitGoogleLogging(argv[0]);
    });
};

}  // namespace lightinfer
