#pragma once

#include "hip/hip_runtime.h"

#include "flashck/core/utils/hip_raii.h"
#include "flashck/core/utils/macros.h"

namespace flashck {

template<typename T>
struct IsSupportedType {
    static constexpr bool value = false;
};

template<>
struct IsSupportedType<float> {
    static constexpr bool value = true;
};

/**
 * @brief Optimized device memory allocator with initialization capabilities
 */
template<typename T>
void DeviceMalloc(
    T**                                                                                           ptr,
    size_t                                                                                        count,
    int                                                                                           init_type = 0,
    typename std::conditional_t<IsSupportedType<T>::value && std::is_floating_point_v<T>, T, int> min_val =
        std::is_floating_point_v<T> ? T(0) : 0,
    typename std::conditional_t<IsSupportedType<T>::value && std::is_floating_point_v<T>, T, int> max_val =
        std::is_floating_point_v<T> ? T(1) : 100);

/**
 * @brief Safely deallocates HIP device memory and nullifies the pointer.
 *
 * @tparam T Type of the elements in the allocated memory. Must match the type used in `hipMalloc`.
 * @param[in,out] ptr Reference to a pointer pointing to HIP device memory.
 *                   After deallocation, the pointer is set to `nullptr`.
 *
 * @note
 * - Idempotent: Safe to call multiple times on the same pointer.
 * - Memory Safety: Ensures no double-free if `ptr` is already `nullptr`.
 * - HIP Integration: Requires the pointer to be allocated via HIP APIs (e.g., `hipMalloc`).
 *
 * @par Example:
 * @code
 * float* d_data = nullptr;
 * HIP_ERROR_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));  // Allocate
 * DeviceFree(d_data);  // Deallocate; d_data becomes nullptr
 * @endcode
 */
template<typename T>
void DeviceFree(T*& ptr);

/**
 * @brief Fills HIP device memory with a specified value asynchronously.
 *
 * @tparam T Data type of the device memory and the fill value.
 * @param[out] devptr Pointer to the device memory to be filled.
 * @param size Number of elements to fill.
 * @param value The value to fill into the device memory.
 * @param stream HIP stream for asynchronous operation (defaults to the null stream).
 *
 * @note
 * - Uses RAII to manage temporary host memory automatically.
 * - The `hipMemcpyAsync` operation requires the host memory (temporary buffer) to remain valid
 *   until the copy completes. If stream synchronization is not guaranteed, consider using
 *   HIP pinned memory instead of the default `std::vector`.
 * - For large datasets, prefer HIP APIs like `hipMemsetAsync` or kernel-based filling for better performance.
 *
 * @par Example:
 * @code
 * float* d_data;
 * HIP_ERROR_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));
 * DeviceFill(d_data, 1024, 3.14f);  // Fill with 3.14f asynchronously
 * HIP_ERROR_CHECK(hipStreamSynchronize(0));  // Optional synchronization
 * @endcode
 */
template<typename T>
void DeviceFill(T* devptr, size_t size, T value, hipStream_t stream = nullptr);

/**
 * @brief Asynchronously copies data from HIP device memory to host memory.
 *
 * @tparam T Data type of the source and destination pointers.
 * @param[out] tgt Pointer to host memory where data will be copied.
 * @param[in] src Pointer to device memory from which data will be copied.
 * @param size Number of elements to copy.
 * @param stream HIP stream for asynchronous operation (default: nullptr).
 *
 * @note
 * - The target host memory (`tgt`) should be pinned (page-locked) for optimal
 *   asynchronous performance. Use `hipHostMalloc` for allocation.
 * - The source device memory (`src`) must be valid and accessible.
 * - Ensure stream synchronization (`hipStreamSynchronize`) or use completion
 *   callbacks to guarantee data is ready before accessing `tgt`.
 */
template<typename T>
void HipD2HCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

/**
 * @brief Asynchronously copies data from host memory to HIP device memory.
 *
 * @tparam T Data type of the source and destination pointers.
 * @param[out] tgt Pointer to device memory where data will be copied.
 * @param[in] src Pointer to host memory from which data will be copied.
 * @param size Number of elements to copy.
 * @param stream HIP stream for asynchronous operation (default: nullptr).
 *
 * @note
 * - The source host memory (`src`) should be pinned (page-locked) for optimal
 *   asynchronous performance. Use `hipHostMalloc` for allocation.
 * - The target device memory (`tgt`) must be properly allocated.
 * - For overlapping compute and data transfer, use non-default streams and ensure
 *   proper synchronization via `hipStreamSynchronize` or event-based callbacks.
 */
template<typename T>
void HipH2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

/**
 * @brief Asynchronously copies data between HIP device memory buffers.
 *
 * @tparam T Data type of the source and destination pointers.
 * @param[out] tgt Pointer to device memory destination.
 * @param[in] src Pointer to device memory source.
 * @param size Number of elements to copy.
 * @param stream HIP stream for asynchronous operation (default: nullptr).
 *
 * @note
 * - Both source (`src`) and target (`tgt`) must reside on the same device context.
 * - For cross-device copies (peer-to-peer), ensure peer access is enabled via `hipDeviceEnablePeerAccess`.
 * - Zero-copy operations have minimal overhead but still require stream synchronization.
 */
template<typename T>
void HipD2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

}  // namespace flashck