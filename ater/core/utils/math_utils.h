#pragma once

namespace ater {

inline int IntegerDivideCeil(int x, int y)
{
    return (x + y - size_t{1}) / y;
}

}  // namespace ater