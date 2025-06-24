#pragma once

#include "gflags/gflags.h"
#include <glog/logging.h>

namespace flashck {

// void PrefixFormatter(std::ostream& s, const google::LogMessage& m, void* data);

void InitGLOG(char** argv);

void InitGflags(int argc, char** argv, bool remove_flags = true);

}  // namespace flashck