#pragma once

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashck/core/utils/enforce.h"

namespace flashck {

/**
 * @class DDim
 * @brief Dynamic dimension with range support for flexible shape handling
 */
class DDim {
public:
    /// Default constructor
    DDim() = default;

    /// Constructor for static dimension
    DDim(int64_t value, std::string name = ""): values_({value, value}), name_(name) {}

    /// Constructor for dynamic dimension with range
    DDim(std::vector<int64_t> values, std::string name = "");

    /// Get minimum dimension value
    int64_t GetLowerBound() const;

    /// Get maximum dimension value
    int64_t GetUpperBound() const;

    /// Get number of possible values
    int64_t GetSize() const;

    /// Get all possible values
    std::vector<int64_t> GetValues() const;

    /// Set dimension values
    void SetValues(const std::vector<int64_t>& values);

    /// Get dimension name
    std::string GetName() const;

    /// Set dimension name
    void SetName(const std::string& name);

    /// Check if dimension is static (single value)
    bool IsStatic() const;

    /// Equality comparison
    bool operator==(const DDim& other) const;

    /// Inequality comparison
    bool operator!=(const DDim& other) const;

    /// Multiplication operator
    DDim operator*(const DDim& other) const;

    /// Addition operator
    DDim operator+(const DDim& other) const;

    /// Subtraction operator
    DDim operator-(const DDim& other) const;

    /// Division operator
    DDim operator/(const DDim& other) const;

    /// Convert to string representation
    std::string ToString();

private:
    std::vector<int64_t> values_;  ///< Dimension values [min, max] or list
    std::string          name_;    ///< Optional dimension name
};

/**
 * @class Shape
 * @brief Multi-dimensional tensor shape with dynamic dimension support
 */
class Shape {
public:
    /// Default constructor
    Shape() = default;

    /// Constructor from vector of dimensions
    Shape(std::vector<DDim> dims, std::string name = "");

    /// Constructor from initializer list
    Shape(std::initializer_list<DDim> dims, std::string name = "");

    /// Copy constructor
    Shape(const Shape& other);

    /// Validate dimension index
    void CheckDims(const int64_t dim) const;

    // Shape query methods
    std::vector<DDim>    ToVector() const;
    std::vector<int64_t> MutableStaticData();
    int64_t              GetNumDim() const;
    DDim                 GetLastDim() const;
    DDim                 GetDim(const int64_t dim) const;
    DDim                 operator[](const int64_t dim) const;

    // Shape modification methods
    void  SetDim(const int64_t dim, const DDim& value);
    Shape InsertDim(const int64_t dim, const DDim& value);
    Shape AppendDim(const DDim& value);
    void  Clear();

    // Comparison operators
    bool operator==(const Shape& other) const;
    bool operator==(const std::initializer_list<DDim>& other) const;
    bool operator!=(const Shape& other) const;
    bool operator!=(const std::initializer_list<DDim>& other) const;

    // Utility methods
    std::tuple<size_t, size_t> GetElementSizeTuple() const;
    std::string                ToString() const;

    /**
     * @brief Compute broadcast-compatible shape for two shapes
     * @param shape_1 First shape to broadcast
     * @param shape_2 Second shape to broadcast
     * @return Tuple of (success, broadcasted_shape)
     */
    static std::tuple<bool, Shape> GetBroadCastMaxShape(const Shape& shape_1, const Shape& shape_2);

private:
    std::vector<DDim> dims_;  ///< Shape dimensions
    std::string       name_;  ///< Optional shape name
};

// Utility functions for vector operations

/**
 * @brief Extract a slice from a vector
 * @param vec Source vector to slice
 * @param start Start index (inclusive)
 * @param end End index (exclusive, -1 for size-based offset)
 * @return Sliced vector
 */
template<typename T>
std::vector<T> SliceVec(const std::vector<T>& vec, int start, int end = -1);

/**
 * @brief Concatenate vector b to the end of vector a
 * @param a Target vector to extend
 * @param b Source vector to append
 */
template<typename T>
void ConcatVec(std::vector<T>& a, const std::vector<T>& b);

}  // namespace flashck