#include "core/utils/hip_utils.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "core/utils/macros.h"

namespace flashck {

// ==============================================================================
// Device Information Functions
// ==============================================================================

std::string GetDeviceName(int device_id)
{
    hipDeviceProp_t props;
    HIP_ERROR_CHECK(hipGetDeviceProperties(&props, device_id));
    const std::string raw_name(props.gcnArchName);

    static const std::unordered_map<std::string_view, std::string_view> device_name_map = {
        {"Ellesmere", "gfx803"},
        {"Baffin", "gfx803"},
        {"RacerX", "gfx803"},
        {"Polaris10", "gfx803"},
        {"Polaris11", "gfx803"},
        {"Tonga", "gfx803"},
        {"Fiji", "gfx803"},
        {"gfx800", "gfx803"},
        {"gfx802", "gfx803"},
        {"gfx804", "gfx803"},
        {"Vega10", "gfx900"},
        {"gfx901", "gfx900"},
        {"10.3.0 Sienna_Cichlid 18", "gfx1030"},
        {"gfx1031", "gfx1030"},
        {"gfx1032", "gfx1030"},
        {"gfx1100", "gfx1100"},
        {"gfx1101", "gfx1100"},
        {"gfx1102", "gfx1100"},
    };

    const size_t           colon_pos = raw_name.find(':');
    const std::string_view name_key =
        colon_pos != std::string::npos ? std::string_view(raw_name).substr(0, colon_pos) : std::string_view(raw_name);

    if (const auto it = device_name_map.find(name_key); it != device_name_map.end()) {
        return std::string(it->second);
    }
    return std::string(name_key);
}

int GetDeviceCount()
{
    int count;
    HIP_ERROR_CHECK(hipGetDeviceCount(&count));
    return count;
}

hipDeviceProp_t GetDeviceProperties(int device_id)
{
    hipDeviceProp_t props;
    HIP_ERROR_CHECK(hipGetDeviceProperties(&props, device_id));
    return props;
}

int GetCurrentDevice()
{
    int device_id;
    HIP_ERROR_CHECK(hipGetDevice(&device_id));
    return device_id;
}

void SetDevice(int device_id)
{
    HIP_ERROR_CHECK(hipSetDevice(device_id));
}

std::pair<size_t, size_t> GetDeviceMemoryInfo(int device_id)
{
    DeviceGuard guard(device_id);
    size_t      free_mem, total_mem;
    HIP_ERROR_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    return {free_mem, total_mem};
}

bool SupportsUnifiedMemory(int device_id)
{
    hipDeviceProp_t props = GetDeviceProperties(device_id);
    return props.managedMemory;
}

// ==============================================================================
// Memory Management Functions
// ==============================================================================

bool IsGpuPointer(const void* ptr)
{
    if (!ptr)
        return false;

    hipPointerAttribute_t attr;
    hipError_t            error = hipPointerGetAttributes(&attr, ptr);

    if (error != hipSuccess) {
        // If query fails, assume it's not a GPU pointer
        return false;
    }

    return attr.type == hipMemoryTypeDevice;
}

bool IsCpuPointer(const void* ptr)
{
    if (!ptr)
        return false;

    hipPointerAttribute_t attr;
    hipError_t            error = hipPointerGetAttributes(&attr, ptr);

    if (error != hipSuccess) {
        // If query fails, assume it's a CPU pointer
        return true;
    }

    return attr.type == hipMemoryTypeHost;
}

hipPointerAttribute_t GetPointerAttributes(const void* ptr)
{
    hipPointerAttribute_t attr;
    HIP_ERROR_CHECK(hipPointerGetAttributes(&attr, ptr));
    return attr;
}

int GetPointerDevice(const void* ptr)
{
    if (!ptr)
        return -1;

    hipPointerAttribute_t attr;
    hipError_t            error = hipPointerGetAttributes(&attr, ptr);

    if (error != hipSuccess || attr.type != hipMemoryTypeDevice) {
        return -1;
    }

    return attr.device;
}

// ==============================================================================
// Stream and Event Management
// ==============================================================================

hipStream_t CreateStream(unsigned int flags)
{
    hipStream_t stream;
    HIP_ERROR_CHECK(hipStreamCreateWithFlags(&stream, flags));
    return stream;
}

void DestroyStream(hipStream_t stream)
{
    if (stream) {
        HIP_ERROR_CHECK(hipStreamDestroy(stream));
    }
}

hipEvent_t CreateEvent(unsigned int flags)
{
    hipEvent_t event;
    HIP_ERROR_CHECK(hipEventCreateWithFlags(&event, flags));
    return event;
}

void DestroyEvent(hipEvent_t event)
{
    if (event) {
        HIP_ERROR_CHECK(hipEventDestroy(event));
    }
}

void StreamSynchronize(hipStream_t stream)
{
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));
}

void DeviceSynchronize()
{
    HIP_ERROR_CHECK(hipDeviceSynchronize());
}

// ==============================================================================
// Error Handling and Utilities
// ==============================================================================

std::string GetErrorString(hipError_t error)
{
    return std::string(hipGetErrorString(error));
}

bool IsHipAvailable()
{
    int        device_count;
    hipError_t error = hipGetDeviceCount(&device_count);
    return error == hipSuccess && device_count > 0;
}

int GetHipRuntimeVersion()
{
    int version;
    HIP_ERROR_CHECK(hipRuntimeGetVersion(&version));
    return version;
}

int GetHipDriverVersion()
{
    int version;
    HIP_ERROR_CHECK(hipDriverGetVersion(&version));
    return version;
}

void PrintDeviceInfo()
{
    const int device_count = GetDeviceCount();

    std::cout << "=== HIP Device Information ===\n";
    std::cout << "Number of devices: " << device_count << "\n";
    std::cout << "Runtime version: " << GetHipRuntimeVersion() << "\n";
    std::cout << "Driver version: " << GetHipDriverVersion() << "\n\n";

    for (int i = 0; i < device_count; ++i) {
        hipDeviceProp_t props      = GetDeviceProperties(i);
        auto [free_mem, total_mem] = GetDeviceMemoryInfo(i);

        std::cout << "Device " << i << ": " << props.name << "\n";
        std::cout << "  Architecture: " << GetDeviceName(i) << "\n";
        std::cout << "  Compute capability: " << props.major << "." << props.minor << "\n";
        std::cout << "  Total memory: " << (total_mem / (1024 * 1024)) << " MB\n";
        std::cout << "  Free memory: " << (free_mem / (1024 * 1024)) << " MB\n";
        std::cout << "  Max threads per block: " << props.maxThreadsPerBlock << "\n";
        std::cout << "  Max block dimensions: (" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", "
                  << props.maxThreadsDim[2] << ")\n";
        std::cout << "  Max grid dimensions: (" << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", "
                  << props.maxGridSize[2] << ")\n";
        std::cout << "  Warp size: " << props.warpSize << "\n";
        std::cout << "  Unified memory: " << (SupportsUnifiedMemory(i) ? "Yes" : "No") << "\n";
        std::cout << "\n";
    }
}

void PrintMemoryUsage(int device_id)
{
    auto [free_mem, total_mem] = GetDeviceMemoryInfo(device_id);
    size_t used_mem            = total_mem - free_mem;

    std::cout << "=== Memory Usage for Device " << device_id << " ===\n";
    std::cout << "Total memory: " << (total_mem / (1024 * 1024)) << " MB\n";
    std::cout << "Used memory: " << (used_mem / (1024 * 1024)) << " MB\n";
    std::cout << "Free memory: " << (free_mem / (1024 * 1024)) << " MB\n";
    std::cout << "Usage: " << std::fixed << std::setprecision(1) << (100.0 * used_mem / total_mem) << "%\n";
}

// ==============================================================================
// RAII Helper Classes
// ==============================================================================

DeviceGuard::DeviceGuard(int device_id):
    device_id_(device_id), previous_device_id_(GetCurrentDevice()), should_restore_(true)
{
    if (device_id_ != previous_device_id_) {
        SetDevice(device_id_);
    }
    else {
        should_restore_ = false;
    }
}

DeviceGuard::~DeviceGuard()
{
    if (should_restore_) {
        try {
            SetDevice(previous_device_id_);
        }
        catch (...) {
            // Ignore exceptions in destructor
        }
    }
}

DeviceGuard::DeviceGuard(DeviceGuard&& other) noexcept:
    device_id_(other.device_id_), previous_device_id_(other.previous_device_id_), should_restore_(other.should_restore_)
{
    other.should_restore_ = false;
}

DeviceGuard& DeviceGuard::operator=(DeviceGuard&& other) noexcept
{
    if (this != &other) {
        if (should_restore_) {
            try {
                SetDevice(previous_device_id_);
            }
            catch (...) {
                // Ignore exceptions
            }
        }

        device_id_            = other.device_id_;
        previous_device_id_   = other.previous_device_id_;
        should_restore_       = other.should_restore_;
        other.should_restore_ = false;
    }
    return *this;
}

StreamGuard::StreamGuard(unsigned int flags): stream_(CreateStream(flags)) {}

StreamGuard::~StreamGuard()
{
    if (stream_) {
        DestroyStream(stream_);
    }
}

StreamGuard::StreamGuard(StreamGuard&& other) noexcept: stream_(other.stream_)
{
    other.stream_ = nullptr;
}

StreamGuard& StreamGuard::operator=(StreamGuard&& other) noexcept
{
    if (this != &other) {
        if (stream_) {
            DestroyStream(stream_);
        }
        stream_       = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void StreamGuard::synchronize()
{
    StreamSynchronize(stream_);
}

EventGuard::EventGuard(unsigned int flags): event_(CreateEvent(flags)) {}

EventGuard::~EventGuard()
{
    if (event_) {
        DestroyEvent(event_);
    }
}

EventGuard::EventGuard(EventGuard&& other) noexcept: event_(other.event_)
{
    other.event_ = nullptr;
}

EventGuard& EventGuard::operator=(EventGuard&& other) noexcept
{
    if (this != &other) {
        if (event_) {
            DestroyEvent(event_);
        }
        event_       = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void EventGuard::record(hipStream_t stream)
{
    HIP_ERROR_CHECK(hipEventRecord(event_, stream));
}

void EventGuard::synchronize()
{
    HIP_ERROR_CHECK(hipEventSynchronize(event_));
}

float EventGuard::elapsedTime(const EventGuard& end_event)
{
    float ms;
    HIP_ERROR_CHECK(hipEventElapsedTime(&ms, event_, end_event.event_));
    return ms;
}

}  // namespace flashck