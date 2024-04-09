#include "ater/core/graph/shape.h"

#include <numeric>

#include "ater/core/utils/enforce.h"

namespace ater {

std::ostream& operator<<(std::ostream& os, const DDim& dim)
{
    os << "(";

    if (dim.GetSize() <= 2) {
        os << "lower:" << dim.GetLowerBound() << ", "
           << "higher:" << dim.GetUpperBound();
    }
    else {
        for (int i = 0; i < dim.GetSize(); i++) {
            os << dim.GetValues()[i] << (i == dim.GetSize() - 1 ? "" : ", ");
        }
    }
    os << ")";
    return os;
}

Shape::Shape(std::vector<DDim> dim): dims_(dim) {}

Shape::Shape(std::initializer_list<DDim> dim): dims_(dim) {}

void Shape::CheckDims(const int dim) const
{
    if (dim > GetNumDim() - 1) {
        ATER_THROW(InvalidArgument("shape {} index out of bounds{}.", dim, dims_.size()));
    }
}

std::vector<DDim> Shape::ToVector() const
{
    return dims_;
}

int Shape::GetNumDim() const
{

    return dims_.size();
}

DDim Shape::GetLastDim() const
{
    return GetDim(GetNumDim() - 1);
}

DDim Shape::GetDim(const int dim) const
{
    CheckDims(dim);
    return dims_[dim];
}

DDim Shape::operator[](const int dim) const
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

std::tuple<int, int> Shape::GetElementSizeTuple() const
{
    if (dims_.size() == 1) {
        return std::make_tuple(dims_[0].GetLowerBound(), dims_[0].GetUpperBound());
    }

    std::tuple<int, int> init = {1, 1};

    std::tuple<int, int> res = std::accumulate(dims_.begin(), dims_.end(), init, [](std::tuple<int, int> a, DDim dim) {
        return std::make_tuple(std::get<0>(a) * dim.GetLowerBound(), std::get<1>(a) * dim.GetUpperBound());
    });
    return res;
}

std::string Shape::ToString() const
{
    std::stringstream ss;
    ss << "(";
    for (int i = 0; i < GetNumDim(); i++) {
        ss << GetDim(i) << (i == GetNumDim() - 1 ? ")" : ",");
    }
    return ss.str();
}

}  // namespace ater