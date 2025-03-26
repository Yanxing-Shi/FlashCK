#pragma once

#include <iomanip>
#include <iostream>

#include <glog/logging.h>

namespace lightinfer {

// void PrefixFormatter(std::ostream& s, const google::LogMessage& m, void* data);

void InitGLOG(char** argv);

}  // namespace lightinfer