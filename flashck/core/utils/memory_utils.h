#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "flashck/core/utils/macros.h"

namespace flashck {

// ==============================================================================
// Type Aliases
// ==============================================================================

// Define ushort alias for consistency
using ushort = unsigned short;

// ==============================================================================
// Type Traits and Concepts
// ==============================================================================

/*!
 * @brief Type trait to determine if a type is supported for GPU operations
 * @tparam T The type to check
 * @note Default implementation returns false for unsupported types
 */
template<typename T>
struct IsSupportedType {
    static constexpr bool value = false;
};

/*!
 * @brief Specialization for float type
 * @note float is fully supported for all GPU operations
 */
template<>
struct IsSupportedType<float> {
    static constexpr bool value = true;
};

/*!
 * @brief Specialization for _Float16 type
 * @note _Float16 is supported for GPU operations on compatible hardware
 */
template<>
struct IsSupportedType<_Float16> {
    static constexpr bool value = true;
};

/*!
 * @brief Specialization for ushort type
 * @note ushort is supported for GPU operations
 */
template<>
struct IsSupportedType<ushort> {
    static constexpr bool value = true;
};

/*!
 * @brief Concept-like alias for supported types
 * @tparam T The type to check
 */
template<typename T>
inline constexpr bool is_supported_type_v = IsSupportedType<T>::value;

// ==============================================================================
// Memory Management Enumerations
// ==============================================================================

/*!
 * @brief Enumeration for memory initialization types
 */
enum class InitType {
    None   = 0,  //!< No initialization (fastest)
    Random = 1,  //!< Random initialization using GPU kernel
    Zero   = 2   //!< Zero initialization using hipMemset
};

// ==============================================================================
// Core Memory Management Functions
// ==============================================================================

/*!
 * @brief Allocate device memory with optional initialization
 * @tparam T Element type (must be a supported type)
 * @param ptr Output pointer to allocated memory
 * @param count Number of elements to allocate
 * @param init_type Initialization type (default: no initialization)
 * @param min_val Minimum value for random initialization (default: 0)
 * @param max_val Maximum value for random initialization (default: 1 for float, 100 for int)
 * @throws std::invalid_argument if parameters are invalid
 * @throws std::runtime_error if HIP allocation fails
 * @note The allocated memory must be freed with DeviceFree
 */
template<typename T>
void DeviceMalloc(T**      ptr,
                  size_t   count,
                  InitType init_type = InitType::None,
                  typename std::conditional_t<is_supported_type_v<T> && std::is_floating_point_v<T>, T, int> min_val =
                      std::is_floating_point_v<T> ? T(0) : 0,
                  typename std::conditional_t<is_supported_type_v<T> && std::is_floating_point_v<T>, T, int> max_val =
                      std::is_floating_point_v<T> ? T(1) : 100);

/*!
 * @brief Free device memory safely
 * @tparam T Element type
 * @param ptr Reference to pointer to be freed (set to nullptr after freeing)
 * @note Safe to call with nullptr
 * @note Pointer is set to nullptr after freeing
 */
template<typename T>
void DeviceFree(T*& ptr);

/*!
 * @brief Fill device memory with a specific value
 * @tparam T Element type
 * @param devptr Device pointer to fill
 * @param size Number of elements to fill
 * @param value Value to fill with
 * @param stream HIP stream for asynchronous operation (default: nullptr)
 * @note Uses optimized kernel for supported types
 * @warning For asynchronous operations, ensure proper stream synchronization
 */
template<typename T>
void DeviceFill(T* devptr, size_t size, T value, hipStream_t stream = nullptr);

// ==============================================================================
// Asynchronous Memory Copy Operations
// ==============================================================================

/*!
 * @brief Asynchronous device-to-host memory copy
 * @tparam T Element type
 * @param tgt Host destination pointer
 * @param src Device source pointer
 * @param size Number of elements to copy
 * @param stream HIP stream for asynchronous operation (default: nullptr)
 * @note Caller must ensure proper stream synchronization
 * @warning Host memory must remain valid until operation completes
 */
template<typename T>
void HipD2HCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

/*!
 * @brief Asynchronous host-to-device memory copy
 * @tparam T Element type
 * @param tgt Device destination pointer
 * @param src Host source pointer
 * @param size Number of elements to copy
 * @param stream HIP stream for asynchronous operation (default: nullptr)
 * @note Caller must ensure proper stream synchronization
 * @warning Host memory must remain valid until operation completes
 */
template<typename T>
void HipH2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

/*!
 * @brief Asynchronous device-to-device memory copy
 * @tparam T Element type
 * @param tgt Device destination pointer
 * @param src Device source pointer
 * @param size Number of elements to copy
 * @param stream HIP stream for asynchronous operation (default: nullptr)
 * @note Caller must ensure proper stream synchronization
 * @note Both pointers must be valid device memory
 */
template<typename T>
void HipD2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

// ==============================================================================
// RAII Memory Management Helper
// ==============================================================================

/*!
 * @brief RAII wrapper for HIP device memory
 * @tparam T Element type
 * @note Automatically frees memory on destruction
 */
template<typename T>
class HipUniquePtr {
public:
    /*!
     * @brief Default constructor (empty pointer)
     */
    HipUniquePtr() noexcept: ptr_(nullptr) {}

    /*!
     * @brief Constructor with raw pointer
     * @param ptr Raw device pointer to manage
     */
    explicit HipUniquePtr(T* ptr) noexcept: ptr_(ptr) {}

    /*!
     * @brief Destructor - automatically frees memory
     */
    ~HipUniquePtr() noexcept
    {
        if (ptr_) {
            DeviceFree(ptr_);
        }
    }

    // Non-copyable
    HipUniquePtr(const HipUniquePtr&)            = delete;
    HipUniquePtr& operator=(const HipUniquePtr&) = delete;

    // Movable
    HipUniquePtr(HipUniquePtr&& other) noexcept: ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }

    HipUniquePtr& operator=(HipUniquePtr&& other) noexcept
    {
        if (this != &other) {
            if (ptr_) {
                DeviceFree(ptr_);
            }
            ptr_       = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /*!
     * @brief Get the raw pointer
     * @return Raw device pointer
     */
    T* get() const noexcept
    {
        return ptr_;
    }

    /*!
     * @brief Release ownership and return raw pointer
     * @return Raw device pointer (caller takes ownership)
     */
    T* release() noexcept
    {
        T* tmp = ptr_;
        ptr_   = nullptr;
        return tmp;
    }

    /*!
     * @brief Reset with new pointer
     * @param ptr New pointer to manage
     */
    void reset(T* ptr = nullptr) noexcept
    {
        if (ptr_) {
            DeviceFree(ptr_);
        }
        ptr_ = ptr;
    }

    /*!
     * @brief Boolean conversion operator
     * @return true if pointer is not nullptr
     */
    explicit operator bool() const noexcept
    {
        return ptr_ != nullptr;
    }

private:
    T* ptr_;
};

/*!
 * @brief Factory function to create HipUniquePtr with allocation
 * @tparam T Element type
 * @param count Number of elements to allocate
 * @param init_type Initialization type
 * @param min_val Minimum value for random initialization
 * @param max_val Maximum value for random initialization
 * @return HipUniquePtr managing the allocated memory
 */
template<typename T>
HipUniquePtr<T>
MakeHipUnique(size_t   count,
              InitType init_type = InitType::None,
              typename std::conditional_t<is_supported_type_v<T> && std::is_floating_point_v<T>, T, int> min_val =
                  std::is_floating_point_v<T> ? T(0) : 0,
              typename std::conditional_t<is_supported_type_v<T> && std::is_floating_point_v<T>, T, int> max_val =
                  std::is_floating_point_v<T> ? T(1) : 100)
{
    T* ptr = nullptr;
    DeviceMalloc(&ptr, count, init_type, min_val, max_val);
    return HipUniquePtr<T>(ptr);
}

}  // namespace flashck