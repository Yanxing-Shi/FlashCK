#pragma once

#include <memory>

#include <hip/hip_runtime.h>

#include "flashck/core/utils/macros.h"

namespace flashck {

/**
 * @brief Deleter for HIP device memory
 * @tparam T Type of the allocated memory
 *
 */
template<typename T>
struct HipMemoryDeleter {
    /**
     * @brief Frees HIP device memory
     * @param ptr Pointer to device memory to free
     * @note No-throw guarantee, null-safe
     */
    void operator()(T* ptr) const
    {
        if (ptr) {
            HIP_ERROR_CHECK(hipFree(ptr));
        }
    }
};

/**
 * @brief Deleter for HIP streams
 */
struct HipStreamDeleter {
    /**
     * @brief Destroys HIP stream
     * @param stream Stream handle to destroy
     * @note No-throw guarantee, null-safe
     */
    void operator()(hipStream_t stream) const
    {
        if (stream) {
            HIP_ERROR_CHECK(hipStreamDestroy(stream));
        }
    }
};

/**
 * @brief Deleter for HIP events
 */
struct HipEventDeleter {
    /**
     * @brief Destroys HIP event
     * @param event Event handle to destroy
     * @note No-throw guarantee, null-safe
     */
    void operator()(hipEvent_t event) const
    {
        if (event) {
            HIP_ERROR_CHECK(hipEventDestroy(event));
        }
    }
};

/**
 * @brief Unique pointer for automatic HIP memory management
 * @tparam T Data type of the allocated memory
 *
 */
template<typename T>
using HipUniquePtr = std::unique_ptr<T, HipMemoryDeleter<T>>;

/**
 * @brief Unique pointer for automatic HIP stream management
 *
 */
using HipStreamUniquePtr = std::unique_ptr<std::remove_pointer_t<hipStream_t>, HipStreamDeleter>;

/**
 * @brief Unique pointer for automatic HIP event management
 *
 */
using HipEventUniquePtr = std::unique_ptr<std::remove_pointer_t<hipEvent_t>, HipEventDeleter>;
}  // namespace flashck