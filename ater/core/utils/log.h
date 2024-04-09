#pragma once

#include <iomanip>
#include <iostream>
#include <mutex>

#include "glog/logging.h"

static std::once_flag g_glog_init_flag;

namespace ater {

inline void PrefixFormatter(std::ostream& s, const google::LogMessageInfo& m, void* data)
{
    // clang-format off
    s << "["
        << "[ATER]" << "("
        << m.severity[0]
        << ")" << ' '
        << std::setw(4) << 1900 + m.time.year()
        << std::setw(2) << 1 + m.time.month()
        << std::setw(2) << m.time.day()
        << ' '
        << std::setw(2) << m.time.hour() << ':'
        << std::setw(2) << m.time.min()  << ':'
        << std::setw(2) << m.time.sec() << "."
        << std::setw(6) << m.time.usec()
        << ' '
        << std::setfill(' ') << std::setw(5)
        << m.thread_id << std::setfill('0')
        << ' '
        << m.filename << ':' << m.line_number << "]";

    // clang-format on
}

inline void InitGLOG(char** argv)
{
    std::call_once(g_glog_init_flag, [&]() {
        FLAGS_logtostderr = true;
        google::InitGoogleLogging(argv[0], &PrefixFormatter);
    });
};

}  // namespace ater