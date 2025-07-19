#pragma once

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>

#include "flashck/core/utils/dtype.h"

namespace flashck {

// ==============================================================================
// Tensor Validation Functions
// ==============================================================================

/*!
 * @brief Comprehensive tensor validation for NaN and Inf detection
 * @tparam T Tensor element type (float, _Float16, ushort, etc.)
 * @param tensor Device pointer to tensor data
 * @param elem_cnt Number of elements in tensor
 * @param tensor_name Name of tensor for error reporting
 * @param stream HIP stream for asynchronous operations
 * @throws std::runtime_error if validation fails
 * @note Uses GPU kernel for high-performance validation
 */
template<typename T>
void ResultChecker(const T* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream = nullptr);

/*!
 * @brief Find maximum value in tensor
 * @tparam T Tensor element type
 * @param result Device pointer to tensor data
 * @param size Number of elements
 * @param stream HIP stream for asynchronous operations
 * @note Optimized for large tensors with RAII memory management
 */
template<typename T>
void CheckMaxVal(const T* result, const int size, hipStream_t stream = nullptr);

/*!
 * @brief Find minimum value in tensor
 * @tparam T Tensor element type
 * @param result Device pointer to tensor data
 * @param size Number of elements
 * @param stream HIP stream for asynchronous operations
 * @note Optimized for large tensors with RAII memory management
 */
template<typename T>
void CheckMinVal(const T* result, const int size, hipStream_t stream = nullptr);

/*!
 * @brief Get tensor statistics (min, max, mean, std)
 * @tparam T Tensor element type
 * @param result Device pointer to tensor data
 * @param size Number of elements
 * @param stream HIP stream for asynchronous operations
 * @return Tuple of (min, max, mean, std)
 */
template<typename T>
std::tuple<float, float, float, float> GetTensorStats(const T* result, const int size, hipStream_t stream = nullptr);

// ==============================================================================
// Output Functions
// ==============================================================================

/*!
 * @brief Print tensor contents to screen with type-specific formatting
 * @tparam T Tensor element type
 * @param result Device pointer to tensor data
 * @param size Number of elements to print
 * @param name Optional name for display
 * @param max_elements Maximum number of elements to display (default: 100)
 * @note Automatically handles type conversions for display
 */
template<typename T>
void PrintToScreen(const T* result, const int size, const std::string& name = "", int max_elements = 100);

/*!
 * @brief Write tensor contents to file with buffered I/O
 * @tparam T Tensor element type
 * @param result Device pointer to tensor data
 * @param size Number of elements to write
 * @param file Output file path
 * @param stream HIP stream for asynchronous operations
 * @param open_mode File open mode (default: std::ios::out)
 * @note Uses RAII and buffered output for optimal performance
 */
template<typename T>
void PrintToFile(const T*           result,
                 const int          size,
                 const char*        file,
                 hipStream_t        stream    = nullptr,
                 std::ios::openmode open_mode = std::ios::out);

/*!
 * @brief Write tensor contents to file with string path
 * @tparam T Tensor element type
 * @param result Device pointer to tensor data
 * @param size Number of elements to write
 * @param filepath Output file path as string
 * @param stream HIP stream for asynchronous operations
 * @param open_mode File open mode (default: std::ios::out)
 */
template<typename T>
void PrintToFile(const T*           result,
                 const int          size,
                 const std::string& filepath,
                 hipStream_t        stream    = nullptr,
                 std::ios::openmode open_mode = std::ios::out);

// ==============================================================================
// Memory Inspection Functions
// ==============================================================================

/*!
 * @brief Inspect device memory region for debugging
 * @tparam T Element type
 * @param ptr Device pointer to inspect
 * @param size Number of elements to inspect
 * @param name Optional name for display
 * @param stream HIP stream for operations
 * @note Provides detailed memory layout information
 */
template<typename T>
void InspectDeviceMemory(const T* ptr, int size, const std::string& name = "", hipStream_t stream = nullptr);

/*!
 * @brief Compare two device tensors element-wise
 * @tparam T Element type
 * @param tensor1 First tensor
 * @param tensor2 Second tensor
 * @param size Number of elements
 * @param tolerance Comparison tolerance
 * @param name Optional name for display
 * @param stream HIP stream for operations
 * @return Number of mismatched elements
 */
template<typename T>
int CompareTensors(const T* tensor1, const T* tensor2, int size, float tolerance = 1e-6f, const std::string& name = "", hipStream_t stream = nullptr);

// ==============================================================================
// Utility Functions
// ==============================================================================

/*!
 * @brief Check if device pointer is valid
 * @param ptr Device pointer to check
 * @return true if pointer is valid device memory
 */
bool IsValidDevicePointer(const void* ptr);

/*!
 * @brief Get readable string representation of tensor type
 * @tparam T Tensor element type
 * @return String representation of type
 */
template<typename T>
constexpr std::string_view GetTypeName();

/*!
 * @brief Format value for display based on type
 * @tparam T Value type
 * @param value Value to format
 * @return Formatted string representation
 */
template<typename T>
std::string FormatValue(const T& value);

}  // namespace flashck