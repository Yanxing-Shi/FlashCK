#pragma once

#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace flashck {

/**
 * @brief Retrieves the list of selected GPU devices
 *
 * @return std::vector<int> Ordered list of unique GPU device IDs. Returns:
 *         - User-specified devices when FLAGS_selected_gpus is set
 *         - All available devices when no selection specified
 *         - Empty vector when:
 *             * Invalid device IDs in FLAGS_selected_gpus
 *             * Specified devices exceed available count
 *
 * @details Selection priorities:
 *          1. Uses FLAGS_selected_gpus if non-empty and valid
 *          2. Falls back to all available devices
 *          3. Returns empty vector for invalid configurations
 *
 * @note Behavior specifics:
 *       - Ignores empty elements in comma-separated list
 *       - Removes duplicate device IDs
 *       - Validates against available device count
 *       - Logs warnings for invalid entries
 */
const std::vector<int> GetSelectedDevices();

/**
 * @brief Retrieves the current active HIP device ID.
 *
 * This function queries the HIP runtime for the currently active device
 * and returns its identifier. The function will throw an exception if
 * the underlying HIP API call fails.
 *
 * @return int The numeric identifier of the current HIP device.
 * @throws std::runtime_error If hipGetDevice returns non-success status.
 */
int GetCurrentDeviceId();

/**
 * @brief Sets the active HIP device and optionally returns the previous device ID.
 *
 * This function performs the following operations:
 * 1. When previous_device_id is not null:
 *    - Stores the current device ID in *previous_device_id
 *    - Only switches device if current device differs from target
 * 2. When previous_device_id is null:
 *    - Unconditionally switches to target device
 *
 * @note This function uses RAII-style error handling through HIP_ERROR_CHECK.
 *       Device switching is thread-local as per HIP specification.
 *
 * @param target_device_id The destination device identifier to activate
 * @param[out] previous_device_id Pointer to store previous device ID (optional)
 * @throws std::runtime_error If any HIP API call returns non-success status.
 *
 * @warning Device IDs are validated implicitly through HIP runtime. Invalid IDs
 *          will trigger hipErrorInvalidDevice exceptions.
 */
void SetDeviceAndGetPrevious(int target_device_id, int* previous_device_id = nullptr);

/**
 * @brief Translates raw device name to standardized architecture code
 *
 * @param device_id Target device identifier
 * @return std::string Normalized architecture name (e.g., "gfx803") or raw prefix
 *
 * @details Parses device properties to extract raw name, then matches against known
 * architectures using a static mapping table. Handles names with colon-separated
 * suffixes by truncating at first colon. Uses string_view for efficient key comparisons
 * while maintaining string return type for interface compatibility.
 */
std::string GetDeviceName(int device_id);

/**
 * @brief Classifies device architecture type.
 *
 * @param device_id The target device identifier.
 * @return std::string Architecture classification ("xdl" or "wmma").
 * @throws std::runtime_error For unsupported legacy architectures.
 *
 * @note The exception message contains both device ID and architecture name
 *       for diagnostic purposes.
 */
std::string GetDeviceArch(int device_id);

/**
 * @brief Determines if a pointer refers to GPU device memory
 *
 * @param[in] ptr The pointer to verify. Can be nullptr.
 *
 * @return true if the pointer points to device-allocated memory
 * @return false for host pointers, invalid pointers, or API errors
 *
 * @details Uses HIP runtime API to inspect pointer attributes:
 * - Validates pointer through hipPointerGetAttributes
 * - Checks memory type field in returned attributes
 * - Handles HIP API errors gracefully
 *
 * @note Thread safety: Uses HIP API calls which are thread-safe
 * @warning For multi-device contexts, ensure correct device is active
 *
 * @par Example:
 * @code
 * void* gpu_ptr = hipMalloc(...);
 * if (IsGpuPointer(gpu_ptr)) {
 *   // Handle device memory
 * }
 * @endcode
 */
bool IsGpuPointer(const void* ptr);

/**
 * @brief Logs GPU memory usage statistics in gigabytes (GB)
 *
 * @param[in] device_name Identifier for distinguishing devices in logs. Can be any
 *                        string-like object (std::string, char*, string_view, etc.).
 *
 * @details This function:
 * 1. Queries HIP runtime for current memory allocation status
 * 2. Calculates used memory as (total - free)
 * 3. Converts values from bytes to GB (1 GB = 1024^3 bytes)
 * 4. Logs formatted output with 2 decimal precision
 *
 * @note Requirements:
 * - HIP runtime must be initialized before calling
 * - Requires C++17 or later for string_view compatibility
 *
 * @warning The string underlying device_name must remain valid during the call
 *
 * @par Example:
 * @code
 * LogGpuMemoryUsage("Primary_GPU");
 * // Output: Primary_GPU Memory - Free: 5.23 GB, Used: 10.77 GB, Total: 16.00 GB
 * @endcode
 *
 * @par Error Handling:
 * Terminates program execution via HIP_ERROR_CHECK macro if HIP API call fails
 */
void LogGpuMemoryUsage(std::string_view device_name);

// HipStreamUniquePtr CreateHipStream()
// {
//     hipStream_t stream;
//     hipError_t  err = hipStreamCreate(&stream);
//     if (err != hipSuccess) {
//         throw std::runtime_error("HIP stream creation failed");
//     }
//     return HipStreamUniquePtr(stream);
// }

// HipEventUniquePtr CreateHipEvent(unsigned flags = hipEventDefault)
// {
//     hipEvent_t event;
//     hipError_t err = hipEventCreateWithFlags(&event, flags);
//     if (err != hipSuccess) {
//         throw std::runtime_error("HIP event creation failed");
//     }
//     return HipEventUniquePtr(event);
// }

}  // namespace flashck