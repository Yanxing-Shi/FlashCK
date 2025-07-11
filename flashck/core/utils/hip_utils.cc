#include "flashck/core/utils/hip_utils.h"

#include <iomanip>

#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/macros.h"

FC_DECLARE_string(selected_gpus);

namespace flashck {

const std::vector<int> GetSelectedDevices()
{
    std::vector<int> devices;

    // Query total number of available HIP devices
    int device_count = 0;
    HIP_ERROR_CHECK(hipGetDeviceCount(&device_count));

    // Early return if no devices found
    if (device_count == 0) {
        LOG(ERROR) << "No available GPU devices detected";
        return devices;
    }

    // Process explicit device selection
    if (!FLAGS_selected_gpus.empty()) {
        std::set<int>      unique_devices;
        std::istringstream stream(FLAGS_selected_gpus);
        std::string        token;
        bool               has_validation_errors = false;

        // Parse comma-separated device list
        while (std::getline(stream, token, ',')) {
            // Remove whitespace from token
            size_t start = token.find_first_not_of(" \t\n\r");
            size_t end   = token.find_last_not_of(" \t\n\r");
            if (start == std::string::npos || end == std::string::npos) {
                continue;  // Skip empty tokens
            }
            token = token.substr(start, end - start + 1);

            if (token.empty())
                continue;

            // Validate numeric format
            if (!std::all_of(token.begin(), token.end(), ::isdigit)) {
                LOG(WARNING) << "Invalid device ID format: " << token;
                has_validation_errors = true;
                continue;
            }

            // Convert and validate device ID range
            try {
                int device_id = std::stoi(token);
                if (device_id < 0 || device_id >= device_count) {
                    LOG(WARNING) << "Device ID out of range: " << device_id << " (Available: 0-" << (device_count - 1)
                                 << ")";
                    has_validation_errors = true;
                }
                else {
                    unique_devices.insert(device_id);
                }
            }
            catch (const std::invalid_argument& e) {
                LOG(WARNING) << "Non-numeric device ID: " << token;
                has_validation_errors = true;
            }
            catch (const std::out_of_range& e) {
                LOG(WARNING) << "Device ID overflow: " << token;
                has_validation_errors = true;
            }
        }

        // Handle validated devices
        if (!unique_devices.empty()) {
            devices.assign(unique_devices.begin(), unique_devices.end());
        }
        else if (has_validation_errors) {
            LOG(WARNING) << "All specified devices are invalid";
        }
    }
    else {
        // Default case: select all available devices
        devices.reserve(device_count);
        for (int i = 0; i < device_count; ++i) {
            devices.emplace_back(i);
        }
    }

    // Final validation and logging
    if (devices.empty()) {
        LOG(ERROR) << "No valid GPU devices selected";
    }
    else {
        std::stringstream ss;
        ss << "Selected GPUs [ ";
        for (int id : devices)
            ss << id << " ";
        ss << "] (" << devices.size() << " devices)";
        LOG(INFO) << ss.str();
    }

    return devices;
}

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