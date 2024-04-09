#pragma once
#include <string>
#include <vector>

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <stdexcept>

#include "ater/core/utils/enforce.h"

namespace ater {

// dynamic dimension
class DDim {
public:
    DDim() = default;

    DDim(int value)
    {
        ATER_ENFORCE_GE(value, 0, Unavailable("Dynamic shape value must > 0, but got {}", value));
        values_ = {value, value};
    }

    // A list of possible values of this dynamic dimension. len(values) must be >= 2.
    // 1. When len(values) == 2, the values are treated as a lower bound and an upper bound.
    // Both upper bound and lower bound are inclusive. This is the default use case.
    // 2. When len(values) > 2, the values are treated as a list of possible values.
    // and the other values are used for internal profiling purpose. This is a legacy use case.
    DDim(std::vector<int> values)
    {
        // value to vector
        ATER_ENFORCE_GE(
            values.size(), 2, Unavailable("Dynamic shape must have at least 1 values, but got {}", values.size()));

        std::sort(values.begin(), values.end());
        ATER_ENFORCE_GE(*values.begin(), 0, Unavailable("Dynamic shape values must > 0, but got {}", *values.begin()));

        values.erase(std::unique(values.begin(), values.end()), values.end());
        if (values.size() == 1) {
            values_ = {*values.begin(), *values.begin()};
        }
        else {
            values_ = values;
        }
    }

    int GetLowerBound() const
    {
        return values_.front();
    }

    int GetUpperBound() const
    {
        return values_.back();
    }

    int GetSize() const
    {
        return values_.size();
    }

    std::vector<int> GetValues() const
    {
        return values_;
    }

    bool operator==(const DDim& other) const
    {
        return values_ == other.values_;
    }

    bool operator!=(const DDim& other) const
    {
        return values_ != other.values_;
    }

protected:
    std::vector<int> values_;
};

std::ostream& operator<<(std::ostream& os, const DDim& dim);

// tensor shape
class Shape {
public:
    Shape() = default;

    Shape(std::vector<DDim> dims);
    Shape(std::initializer_list<DDim> dim);

    void CheckDims(const int dim) const;

    std::vector<DDim> ToVector() const;
    int               GetNumDim() const;
    DDim              GetLastDim() const;

    DDim                 GetDim(const int dim) const;
    DDim                 operator[](const int dim) const;
    bool                 operator==(const Shape& other) const;
    bool                 operator==(const std::initializer_list<DDim>& other) const;
    bool                 operator!=(const Shape& other) const;
    bool                 operator!=(const std::initializer_list<DDim>& other) const;
    std::tuple<int, int> GetElementSizeTuple() const;
    std::string          ToString() const;

private:
    std::vector<DDim> dims_;
};

}  // namespace ater