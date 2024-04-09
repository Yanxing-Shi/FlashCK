#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include "ater/core/utils/log.h"

namespace ater {

// Gemm
enum class DataLayout {
    // Gemm
    UNDEFINED  = 0,
    RCR        = 0,
    RRR        = 1,
    ALL_LAYOUT = 3,
};

inline DataLayout StringToDataLayout(const std::string& str)
{

    std::string str_upper = str;
    std::transform(str_upper.begin(), str_upper.end(), str_upper.begin(), ::toupper);
    if (str_upper == "RCR") {
        return DataLayout::RCR;
    }
    else if (str_upper == "RRR") {
        return DataLayout::RRR;
    }
    else if (str_upper == "ALL_LAYOUT") {
        return DataLayout::ALL_LAYOUT;
    }
    else {
        return DataLayout::UNDEFINED;
    }
}

inline const std::string DataLayoutToString(const DataLayout& layout)
{
    switch (layout) {
        case DataLayout::RCR:
            return "RCR";
        case DataLayout::RRR:
            return "RRR";
        case DataLayout::ALL_LAYOUT:
            return "ALL_LAYOUT";
        default:
            LOG(ERROR) << "DataLayout " << static_cast<int>(layout) << " is not supported.";
            return "UNDEFINED";
    }
}

inline std::ostream& operator<<(std::ostream& os, const DataLayout& layout)
{
    os << DataLayoutToString(layout);
    return os;
}

}  // namespace ater