#include "lightinfer/core/graph/shape.h"

#include <numeric>

#include "lightinfer/core/utils/enforce.h"

namespace lightinfer {

Shape::Shape(std::vector<DDim> dim, std::string name): dims_(dim), name_(name) {}

Shape::Shape(std::initializer_list<DDim> dim, std::string name): dims_(dim), name_(name) {}

void Shape::CheckDims(const int64_t dim) const
{
    if (dim > GetNumDim() - 1) {
        LI_THROW(InvalidArgument("shape {} index out of bounds{}.", dim, dims_.size()));
    }
}

std::vector<DDim> Shape::ToVector() const
{
    return dims_;
}

std::vector<int64_t> Shape::MutableStaticData()
{
    std::vector<int64_t> shape_data(this->GetNumDim());

    for (int64_t i = 0; i < this->GetNumDim(); i++) {
        shape_data[i] = this->GetDim(i).GetValues()[0];
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

    if (dim > GetNumDim()) {
        LI_THROW(InvalidArgument("shape {} index out of bounds{}.", dim, dims_.size()));
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
    return !(this->operator==(other));
}

bool Shape::operator==(const std::initializer_list<DDim>& other) const
{
    return dims_.size() == other.size() && std::equal(std::begin(dims_), std::end(dims_), std::begin(other));
}

bool Shape::operator!=(const std::initializer_list<DDim>& other) const
{
    return !(this->operator==(other));
}

std::tuple<size_t, size_t> Shape::GetElementSizeTuple() const
{
    if (dims_.size() == 1) {
        return std::make_tuple(dims_[0].GetLowerBound(), dims_[0].GetUpperBound());
    }

    std::tuple<size_t, size_t> init = {1, 1};

    std::tuple<size_t, size_t> res =
        std::accumulate(dims_.begin(), dims_.end(), init, [](std::tuple<size_t, size_t> a, DDim dim) {
            return std::make_tuple(std::get<0>(a) * dim.GetLowerBound(), std::get<1>(a) * dim.GetUpperBound());
        });

    return res;
}

std::string Shape::ToString() const
{
    std::stringstream ss;
    ss << "(";
    for (int64_t i = 0; i < GetNumDim(); i++) {
        ss << GetDim(i).ToString() << (i == GetNumDim() - 1 ? ")" : ",");
    }
    return ss.str();
}

}  // namespace lightinfer