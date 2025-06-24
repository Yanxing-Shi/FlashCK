#pragma once
#include <string>
#include <vector>

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <stdexcept>

#include "flashck/core/utils/enforce.h"

#include "flashck/core/graph/shape_utils.h"

namespace flashck {

// dynamic dimension
class DDim {
public:
    DDim() = default;

    DDim(int64_t value, std::string name = ""): values_({value, value}), name_(name) {}

    DDim(std::vector<int64_t> values, std::string name = ""): name_(name)
    {
        // value to vector
        LI_ENFORCE_GE(
            values.size(), 1, Unavailable("Dynamic shape must have at least 1 values, but got {}", values.size()));

        std::sort(values.begin(), values.end(), [](int64_t a, int64_t b) { return std::abs(a) < std::abs(b); });
        // LI_ENFORCE_GE(*values.begin(), 0, Unavailable("Dynamic shape values must > 0, but got {}",
        // *values.begin()));

        values.erase(std::unique(values.begin(), values.end()), values.end());
        if (values.size() == 1) {
            values_ = {*values.begin(), *values.begin()};
        }
        else {
            values_ = values;
        }

        std::transform(values.begin(), values.end(), values_.begin(), [](int64_t v) { return std::abs(v); });
    }

    int64_t GetLowerBound() const
    {
        return values_.front();
    }

    int64_t GetUpperBound() const
    {
        return values_.back();
    }

    int64_t GetSize() const
    {
        return values_.size();
    }

    std::vector<int64_t> GetValues() const
    {
        return values_;
    }

    void SetValues(const std::vector<int64_t>& values)
    {
        values_ = values;
    }

    std::string GetName() const
    {
        return name_;
    }

    void SetName(const std::string& name)
    {
        name_ = name;
    }

    bool IsStatic() const
    {
        return values_.front() == values_.back();
    }

    bool operator==(const DDim& other) const
    {
        return values_ == other.values_;
    }

    bool operator!=(const DDim& other) const
    {
        return values_ != other.values_;
    }

    DDim operator*(const DDim& other) const
    {
        std::vector<int64_t> new_values = {values_.front() * other.values_.front(),
                                           values_.back() * other.values_.back()};

        return DDim(new_values);
    }

    DDim operator+(const DDim& other) const
    {
        std::vector<int64_t> new_values = {values_.front() + other.values_.front(),
                                           values_.back() + other.values_.back()};

        return DDim(new_values);
    }

    DDim operator-(const DDim& other) const
    {
        std::vector<int64_t> new_values = {values_.front() - other.values_.front(),
                                           values_.back() - other.values_.back()};

        return DDim(new_values);
    }

    DDim operator/(const DDim& other) const
    {
        int64_t int64_one = 1;

        int64_t min_value = std::max(
            int64_one, static_cast<int64_t>(std::floor(values_.front() / std::max(int64_one, other.values_.front()))));
        int64_t max_value = static_cast<int64_t>(std::ceil(values_.back() / std::max(int64_one, other.values_.back())));

        return DDim({min_value, max_value});
    }

    std::string ToString()
    {
        std::stringstream ss;
        ss << "(";

        if (values_.size() <= 2) {
            ss << "lower:" << values_[0] << ", " << "higher:" << values_[1];
        }
        else {
            for (int64_t i = 0; i < values_.size(); i++) {
                ss << values_[i] << (i == values_.size() - 1 ? "" : ", ");
            }
        }
        ss << ")";
        return ss.str();
    }

protected:
    std::vector<int64_t> values_;
    std::string          name_;
};

// tensor shape
class Shape {
public:
    Shape() = default;

    Shape(std::vector<DDim> dims, std::string name = "");
    Shape(std::initializer_list<DDim> dim, std::string name = "");

    // copy constuctor
    Shape(const Shape& other)
    {
        dims_ = other.dims_;
        name_ = other.name_;
    }

    void CheckDims(const int64_t dim) const;

    std::vector<DDim>    ToVector() const;
    std::vector<int64_t> MutableStaticData();
    int64_t              GetNumDim() const;
    DDim                 GetLastDim() const;
    void                 SetDim(const int64_t dim, const DDim& value);

    Shape InsertDim(const int64_t dim, const DDim& value);
    Shape AppendDim(const DDim& value);

    DDim                       GetDim(const int64_t dim) const;
    DDim                       operator[](const int64_t dim) const;
    bool                       operator==(const Shape& other) const;
    bool                       operator==(const std::initializer_list<DDim>& other) const;
    bool                       operator!=(const Shape& other) const;
    bool                       operator!=(const std::initializer_list<DDim>& other) const;
    std::tuple<size_t, size_t> GetElementSizeTuple() const;
    std::string                ToString() const;

    void Clear()
    {
        dims_.clear();
    }

    static std::tuple<bool, Shape> GetBroadCastMaxShape(const Shape& shape_1, const Shape& shape_2)
    {
        int64_t           min_len = std::min(shape_1.GetNumDim(), shape_2.GetNumDim());
        std::vector<DDim> result_shape =
            shape_1.GetNumDim() > shape_2.GetNumDim() ? shape_1.ToVector() : shape_2.ToVector();

        for (int idx = min_len - 1; idx >= 0; idx--) {
            auto dim_1 = shape_1.GetDim(idx);
            auto dim_2 = shape_2.GetDim(idx);
            if (dim_1 == dim_2) {
                result_shape[idx] = dim_1;
                continue;
            }
            if (dim_1 == DDim(1)) {
                result_shape[idx] = dim_2;
            }
            else if (dim_2 == DDim(1)) {
                result_shape[idx] = dim_1;
            }
            else {
                return std::make_tuple(false, Shape());
            }
        }

        return std::make_tuple(true, Shape(result_shape));
    }

private:
    std::vector<DDim> dims_;
    std::string       name_;
};

}  // namespace flashck