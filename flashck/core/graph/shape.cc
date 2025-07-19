#include "flashck/core/graph/shape.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include "flashck/core/utils/enforce.h"

namespace flashck {

// DDim class implementations

DDim::DDim(std::vector<int64_t> values, std::string name): name_(name)
{
    FC_ENFORCE_GE(values.size(), 1, Unavailable("Dynamic shape must have at least 1 value, got {}", values.size()));

    // Sort and deduplicate values
    std::sort(values.begin(), values.end(), [](int64_t a, int64_t b) { return std::abs(a) < std::abs(b); });
    values.erase(std::unique(values.begin(), values.end()), values.end());

    // Store as range [min, max]
    if (values.size() == 1) {
        values_ = {*values.begin(), *values.begin()};
    }
    else {
        values_ = values;
    }

    // Ensure all values are positive
    std::transform(values.begin(), values.end(), values_.begin(), [](int64_t v) { return std::abs(v); });
}

int64_t DDim::GetLowerBound() const
{
    return values_.front();
}

int64_t DDim::GetUpperBound() const
{
    return values_.back();
}

int64_t DDim::GetSize() const
{
    return values_.size();
}

std::vector<int64_t> DDim::GetValues() const
{
    return values_;
}

void DDim::SetValues(const std::vector<int64_t>& values)
{
    values_ = values;
}

std::string DDim::GetName() const
{
    return name_;
}

void DDim::SetName(const std::string& name)
{
    name_ = name;
}

bool DDim::IsStatic() const
{
    return values_.front() == values_.back();
}

bool DDim::operator==(const DDim& other) const
{
    return values_ == other.values_;
}

bool DDim::operator!=(const DDim& other) const
{
    return values_ != other.values_;
}

DDim DDim::operator*(const DDim& other) const
{
    return DDim({values_.front() * other.values_.front(), values_.back() * other.values_.back()});
}

DDim DDim::operator+(const DDim& other) const
{
    return DDim({values_.front() + other.values_.front(), values_.back() + other.values_.back()});
}

DDim DDim::operator-(const DDim& other) const
{
    return DDim({values_.front() - other.values_.front(), values_.back() - other.values_.back()});
}

DDim DDim::operator/(const DDim& other) const
{
    int64_t min_val = std::max(
        static_cast<int64_t>(1),
        static_cast<int64_t>(std::floor(values_.front() / std::max(static_cast<int64_t>(1), other.values_.front()))));
    int64_t max_val =
        static_cast<int64_t>(std::ceil(values_.back() / std::max(static_cast<int64_t>(1), other.values_.back())));
    return DDim({min_val, max_val});
}

std::string DDim::ToString()
{
    std::stringstream ss;
    ss << "(";
    if (values_.size() <= 2) {
        ss << "min:" << values_[0] << ", max:" << values_[1];
    }
    else {
        for (size_t i = 0; i < values_.size(); i++) {
            ss << values_[i] << (i == values_.size() - 1 ? "" : ", ");
        }
    }
    ss << ")";
    return ss.str();
}

// Shape class implementations

Shape::Shape(std::vector<DDim> dims, std::string name): dims_(dims), name_(name) {}

Shape::Shape(std::initializer_list<DDim> dims, std::string name): dims_(dims), name_(name) {}

Shape::Shape(const Shape& other): dims_(other.dims_), name_(other.name_) {}

void Shape::CheckDims(const int64_t dim) const
{
    if (dim >= GetNumDim() || dim < 0) {
        FC_THROW(InvalidArgument("Shape index {} out of bounds for shape with {} dimensions", dim, dims_.size()));
    }
}

std::vector<DDim> Shape::ToVector() const
{
    return dims_;
}

std::vector<int64_t> Shape::MutableStaticData()
{
    std::vector<int64_t> shape_data(GetNumDim());
    for (int64_t i = 0; i < GetNumDim(); i++) {
        shape_data[i] = GetDim(i).GetValues()[0];
    }
    return shape_data;
}

int64_t Shape::GetNumDim() const
{
    return dims_.size();
}

DDim Shape::GetLastDim() const
{
    return GetDim(GetNumDim() - 1);
}

DDim Shape::GetDim(const int64_t dim) const
{
    CheckDims(dim);
    return dims_[dim];
}

void Shape::SetDim(const int64_t dim, const DDim& value)
{
    CheckDims(dim);
    dims_[dim] = value;
}

Shape Shape::InsertDim(const int64_t dim, const DDim& value)
{
    if (GetNumDim() == 0) {
        dims_.push_back(value);
        return *this;
    }

    if (dim > GetNumDim() || dim < 0) {
        FC_THROW(InvalidArgument("Insert index {} out of bounds for shape with {} dimensions", dim, dims_.size()));
    }

    dims_.insert(dims_.begin() + dim, value);
    return *this;
}

Shape Shape::AppendDim(const DDim& value)
{
    return InsertDim(GetNumDim(), value);
}

DDim Shape::operator[](const int64_t dim) const
{
    CheckDims(dim);
    return dims_[dim];
}

bool Shape::operator==(const Shape& other) const
{
    return dims_ == other.dims_;
}

bool Shape::operator!=(const Shape& other) const
{
    return !(*this == other);
}

bool Shape::operator==(const std::initializer_list<DDim>& other) const
{
    return dims_.size() == other.size() && std::equal(dims_.begin(), dims_.end(), other.begin());
}

bool Shape::operator!=(const std::initializer_list<DDim>& other) const
{
    return !(*this == other);
}

std::tuple<size_t, size_t> Shape::GetElementSizeTuple() const
{
    if (dims_.empty()) {
        return std::make_tuple(0, 0);
    }

    if (dims_.size() == 1) {
        return std::make_tuple(dims_[0].GetLowerBound(), dims_[0].GetUpperBound());
    }

    // Compute total element count as product of all dimensions
    std::tuple<size_t, size_t> result = {1, 1};
    for (const auto& dim : dims_) {
        std::get<0>(result) *= dim.GetLowerBound();
        std::get<1>(result) *= dim.GetUpperBound();
    }

    return result;
}

std::string Shape::ToString() const
{
    std::stringstream ss;
    ss << "(";
    for (int64_t i = 0; i < GetNumDim(); i++) {
        ss << GetDim(i).ToString() << (i == GetNumDim() - 1 ? ")" : ", ");
    }
    return ss.str();
}

void Shape::Clear()
{
    dims_.clear();
}

std::tuple<bool, Shape> Shape::GetBroadCastMaxShape(const Shape& shape_1, const Shape& shape_2)
{
    int64_t           min_len = std::min(shape_1.GetNumDim(), shape_2.GetNumDim());
    std::vector<DDim> result_shape =
        shape_1.GetNumDim() > shape_2.GetNumDim() ? shape_1.ToVector() : shape_2.ToVector();

    // Check broadcast compatibility from rightmost dimensions
    for (int idx = min_len - 1; idx >= 0; idx--) {
        auto dim_1 = shape_1.GetDim(idx);
        auto dim_2 = shape_2.GetDim(idx);

        if (dim_1 == dim_2) {
            result_shape[idx] = dim_1;
        }
        else if (dim_1 == DDim(1)) {
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

// Template function implementations

template<typename T>
std::vector<T> SliceVec(const std::vector<T>& vec, int start, int end)
{
    // Handle negative end index as offset from size
    if (end < 0) {
        end = static_cast<int>(vec.size()) + end + 1;
    }
    else {
        end = std::min(end, static_cast<int>(vec.size()));
    }

    // Validate slice parameters
    if (start > end || start < 0 || start >= static_cast<int>(vec.size())) {
        FC_THROW(InvalidArgument(
            "SliceVec: invalid slice parameters [{}, {}) for vector of size {}", start, end, vec.size()));
    }

    // Handle single element case
    if (start == end - 1) {
        return {vec[start]};
    }

    return std::vector<T>(vec.begin() + start, vec.begin() + end);
}

template<typename T>
void ConcatVec(std::vector<T>& a, const std::vector<T>& b)
{
    a.reserve(a.size() + b.size());
    a.insert(a.end(), b.begin(), b.end());
}

// Explicit template instantiations for common types
template std::vector<int>     SliceVec(const std::vector<int>&, int, int);
template std::vector<int64_t> SliceVec(const std::vector<int64_t>&, int, int);
template std::vector<float>   SliceVec(const std::vector<float>&, int, int);
template std::vector<double>  SliceVec(const std::vector<double>&, int, int);
template std::vector<DDim>    SliceVec(const std::vector<DDim>&, int, int);

template void ConcatVec(std::vector<int>&, const std::vector<int>&);
template void ConcatVec(std::vector<int64_t>&, const std::vector<int64_t>&);
template void ConcatVec(std::vector<float>&, const std::vector<float>&);
template void ConcatVec(std::vector<double>&, const std::vector<double>&);
template void ConcatVec(std::vector<DDim>&, const std::vector<DDim>&);

}  // namespace flashck