#pragma once

namespace lightinfer {

inline int IntegerDivideCeil(int x, int y)
{
    return (x + y - size_t{1}) / y;
}

}  // namespace lightinfer