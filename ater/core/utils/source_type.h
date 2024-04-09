#pragma once

#include <iostream>

namespace ater {
// kernel source type
enum class SourceType {
    CK             = 0,  // intertal kernel
    HIP            = 1,  // user write kernel
    UNKNOWN_SOURCE = 2,
};

inline std::ostream& operator<<(std::ostream& os, const SourceType& source_type)
{
    switch (source_type) {
        case SourceType::CK:
            os << "CK";
            break;
        case SourceType::HIP:
            os << "HIP";
            break;
        default:
            // LOG(ERROR) << "SourceType " << static_cast<int>(source_type) << " is not supported.";
            os << "UNDEFINED";
    }
    return os;
}
}  // namespace ater