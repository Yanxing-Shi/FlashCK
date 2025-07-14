#include "flashck/core/utils/hip_utils.h"

#include <iomanip>

#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/macros.h"

namespace flashck {

int GetCurrentDeviceId()
{
    int device_id;
    HIP_ERROR_CHECK(hipGetDevice(&device_id));
    return device_id;
}

void SetDeviceAndGetPrevious(int target_device_id, int* previous_device_id)
{
    if (previous_device_id != nullptr) {
        *previous_device_id = GetCurrentDeviceId();
        if (*previous_device_id != target_device_id) {
            HIP_ERROR_CHECK(hipSetDevice(target_device_id));
        }
    }
    else {
        HIP_ERROR_CHECK(hipSetDevice(target_device_id));
    }
}

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
    };

    const size_t           colon_pos = raw_name.find(':');
    const std::string_view name_key =
        colon_pos != std::string::npos ? std::string_view(raw_name).substr(0, colon_pos) : std::string_view(raw_name);

    if (const auto it = device_name_map.find(name_key); it != device_name_map.end()) {
        return std::string(it->second);
    }
    return std::string(name_key);
}

bool IsGpuPointer(const void* ptr)
{
    hipPointerAttribute_t attr;
    HIP_ERROR_CHECK(hipPointerGetAttributes(&attr, ptr));

    return attr.type == hipMemoryTypeDevice;
}

void LogGpuMemoryUsage(std::string_view device_name)
{
    constexpr float BYTES_TO_GB = 1.0f / (1024 * 1024 * 1024);  // 2^30 bytes/GB

    size_t free_bytes  = 0;
    size_t total_bytes = 0;

    // HIP API call with error checking
    HIP_ERROR_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

    // Convert to GB with single-precision arithmetic
    const float free_gb  = free_bytes * BYTES_TO_GB;
    const float total_gb = total_bytes * BYTES_TO_GB;
    const float used_gb  = total_gb - free_gb;

    // Log formatted output using stream manipulators
    LOG(INFO) << device_name << " Memory - Free: " << std::fixed << std::setprecision(2) << free_gb
              << " GB, Used: " << used_gb << " GB, Total: " << total_gb << " GB";
}

}  // namespace flashck