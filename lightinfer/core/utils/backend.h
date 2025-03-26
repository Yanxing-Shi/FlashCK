#pragma once

#include <string>

#include "lightinfer/core/utils/enforce.h"

namespace lightinfer {

enum class BackendType {
    CPU        = 0,
    CPU_PINNED = 1,
    GPU        = 2,
    UNKNOWN    = 3,
};

// Get backend type from string
inline BackendType BackendTypeFromString(const std::string& str)
{
    if (str == "CPU") {
        return BackendType::CPU;
    }
    else if (str == "GPU") {
        return BackendType::GPU;
    }
    else if (str == "CPU_PINNED") {
        return BackendType::CPU_PINNED;
    }
    else {
        LI_THROW(Unavailable("BackendType {} is not supported.", str));
    }
}

// Get backend type string
inline std::string BackendTypeToString(const BackendType& type)
{
    switch (type) {
        case BackendType::CPU:
            return "CPU";
        case BackendType::GPU:
            return "GPU";
        case BackendType::CPU_PINNED:
            return "CPU_PINNED";
        default:
            return "Unknown";
    }
}

}  // namespace lightinfer