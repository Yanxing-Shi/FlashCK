#pragma once

#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace flashck {

// ==============================================================================
// Device Information Functions
// ==============================================================================

/**
 * @brief Get the device name for a specific device
 * @param device_id Device ID (default: 0)
 * @return Device name string
 * @throws std::runtime_error if device query fails
 */
std::string GetDeviceName(int device_id = 0);

/**
 * @brief Get the total number of HIP devices
 * @return Number of available HIP devices
 */
int GetDeviceCount();

/**
 * @brief Get detailed device properties
 * @param device_id Device ID
 * @return Device properties structure
 * @throws std::runtime_error if device query fails
 */
hipDeviceProp_t GetDeviceProperties(int device_id = 0);

/**
 * @brief Get current device ID
 * @return Current device ID
 */
int GetCurrentDevice();

/**
 * @brief Set current device
 * @param device_id Device ID to set
 */
void SetDevice(int device_id);

/**
 * @brief Get device memory information
 * @param device_id Device ID
 * @return Pair of (free_memory, total_memory) in bytes
 */
std::pair<size_t, size_t> GetDeviceMemoryInfo(int device_id = 0);

/**
 * @brief Check if device supports unified memory
 * @param device_id Device ID
 * @return true if unified memory is supported
 */
bool SupportsUnifiedMemory(int device_id = 0);

// ==============================================================================
// Memory Management Functions
// ==============================================================================

/**
 * @brief Check if a pointer is a GPU pointer
 * @param ptr Pointer to check
 * @return true if pointer is on GPU memory
 */
bool IsGpuPointer(const void* ptr);

/**
 * @brief Check if a pointer is a CPU pointer
 * @param ptr Pointer to check
 * @return true if pointer is on CPU memory
 */
bool IsCpuPointer(const void* ptr);

/**
 * @brief Get pointer attributes
 * @param ptr Pointer to query
 * @return Pointer attributes structure
 */
hipPointerAttribute_t GetPointerAttributes(const void* ptr);

/**
 * @brief Get the device ID that owns a pointer
 * @param ptr Pointer to query
 * @return Device ID, or -1 if not a device pointer
 */
int GetPointerDevice(const void* ptr);

// ==============================================================================
// Stream and Event Management
// ==============================================================================

/**
 * @brief Create a HIP stream with proper error handling
 * @param flags Stream creation flags
 * @return Created stream
 */
hipStream_t CreateStream(unsigned int flags = hipStreamDefault);

/**
 * @brief Destroy a HIP stream safely
 * @param stream Stream to destroy
 */
void DestroyStream(hipStream_t stream);

/**
 * @brief Create a HIP event with proper error handling
 * @param flags Event creation flags
 * @return Created event
 */
hipEvent_t CreateEvent(unsigned int flags = hipEventDefault);

/**
 * @brief Destroy a HIP event safely
 * @param event Event to destroy
 */
void DestroyEvent(hipEvent_t event);

/**
 * @brief Synchronize a stream
 * @param stream Stream to synchronize (nullptr for default stream)
 */
void StreamSynchronize(hipStream_t stream = nullptr);

/**
 * @brief Synchronize the device
 */
void DeviceSynchronize();

// ==============================================================================
// Error Handling and Utilities
// ==============================================================================

/**
 * @brief Get error string from HIP error code
 * @param error HIP error code
 * @return Error message string
 */
std::string GetErrorString(hipError_t error);

/**
 * @brief Check if HIP runtime is available
 * @return true if HIP runtime is available
 */
bool IsHipAvailable();

/**
 * @brief Get HIP runtime version
 * @return HIP runtime version
 */
int GetHipRuntimeVersion();

/**
 * @brief Get HIP driver version
 * @return HIP driver version
 */
int GetHipDriverVersion();

/**
 * @brief Print device information for all devices
 */
void PrintDeviceInfo();

/**
 * @brief Print memory usage for a specific device
 * @param device_id Device ID
 */
void PrintMemoryUsage(int device_id = 0);

// ==============================================================================
// RAII Helper Classes
// ==============================================================================

/**
 * @brief RAII wrapper for device context management
 */
class DeviceGuard {
public:
    explicit DeviceGuard(int device_id);
    ~DeviceGuard();

    // Non-copyable, movable
    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&& other) noexcept;
    DeviceGuard& operator=(DeviceGuard&& other) noexcept;

    int get() const
    {
        return device_id_;
    }

private:
    int  device_id_;
    int  previous_device_id_;
    bool should_restore_;
};

/**
 * @brief RAII wrapper for HIP streams
 */
class StreamGuard {
public:
    StreamGuard(unsigned int flags = hipStreamDefault);
    ~StreamGuard();

    // Non-copyable, movable
    StreamGuard(const StreamGuard&)            = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    StreamGuard(StreamGuard&& other) noexcept;
    StreamGuard& operator=(StreamGuard&& other) noexcept;

    hipStream_t get() const
    {
        return stream_;
    }
    operator hipStream_t() const
    {
        return stream_;
    }

    void synchronize();

private:
    hipStream_t stream_;
};

/**
 * @brief RAII wrapper for HIP events
 */
class EventGuard {
public:
    EventGuard(unsigned int flags = hipEventDefault);
    ~EventGuard();

    // Non-copyable, movable
    EventGuard(const EventGuard&)            = delete;
    EventGuard& operator=(const EventGuard&) = delete;
    EventGuard(EventGuard&& other) noexcept;
    EventGuard& operator=(EventGuard&& other) noexcept;

    hipEvent_t get() const
    {
        return event_;
    }
    operator hipEvent_t() const
    {
        return event_;
    }

    void  record(hipStream_t stream = nullptr);
    void  synchronize();
    float elapsedTime(const EventGuard& end_event);

private:
    hipEvent_t event_;
};

}  // namespace flashck