#include "flashck/core/utils/hip_utils.h"

#include <iomanip>

#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/macros.h"

FC_DECLARE_string(selected_gpus);

namespace flashck {

/**
 * @brief Retrieves the list of selected GPU device IDs based on runtime flags and system availability.
 *
 * @details This function selects GPU devices according to the following logic:
 * 1. Queries the total number of available HIP-capable GPU devices
 * 2. If the FLAGS_selected_gpus string is non-empty:
 *    - Parses comma-separated device IDs
 *    - Validates numeric format and device ID range
 *    - Filters duplicates and invalid entries
 *    - Falls back to all devices if no valid entries found
 * 3. If FLAGS_selected_gpus is empty, selects all available devices
 *
 * @return const std::vector<int> Vector containing valid GPU device IDs. Returns empty vector if:
 *         - No GPU devices are detected
 *         - All specified devices in FLAGS_selected_gpus are invalid
 *         - System HIP API reports zero available devices
 *
 * @note The device selection string (FLAGS_selected_gpus) should use comma-separated integer values
 *       representing valid device IDs (0-based index). Invalid entries will generate warnings but
 *       allow selection of other valid devices.
 *
 * @warning The following conditions trigger error logging:
 *          - No GPU devices detected (LOG ERROR)
 *          - All specified devices invalid (LOG WARNING)
 *          - Non-numeric device IDs in input (LOG WARNING)
 *          - Device IDs exceeding available range (LOG WARNING)
 *
 * @par Example Usage:
 * FLAGS_selected_gpus = "0,2,3" selects devices 0, 2, and 3 if they exist
 * FLAGS_selected_gpus = "invalid,5" with 3 devices available selects device 5 (invalid) generates warnings
 */
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
int GetCurrentDeviceId()
{
    int device_id;
    HIP_ERROR_CHECK(hipGetDevice(&device_id));
    return device_id;
}

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

/**
 * @brief Determines XDL architecture support status
 *
 * @param device_name Normalized architecture name from GetDeviceName()
 * @return true If device matches known XDL-capable architectures:
 *              - gfx908 (AMD CDNA1)
 *              - gfx90a (AMD CDNA2)
 *              - gfx940/941/942 (AMD CDNA3)
 * @return false For all other architectures
 */
bool IsXdlSupported(std::string_view device_name)
{
    return device_name == "gfx908" || device_name == "gfx90a" || device_name == "gfx940" || device_name == "gfx941"
           || device_name == "gfx942";
}

/**
 * @brief Verifies WMMA instruction set support
 *
 * @param device_name Normalized architecture name
 * @return true If device matches RDNA3 WMMA-enabled architectures:
 *              - gfx1100 (Navi 31)
 *              - gfx1101 (Navi 32)
 * @return false For unsupported architectures
 */
bool IsWmmaSupported(std::string_view device_name)
{
    return device_name == "gfx1100" || device_name == "gfx1101";
}

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
std::string GetDeviceArch(int device_id)
{
    const std::string device_name = GetDeviceName(device_id);

    if (IsXdlSupported(device_name)) {
        return "xdl";
    }
    if (IsWmmaSupported(device_name)) {
        return "wmma";
    }

    // Get raw device name for error reporting
    hipDeviceProp_t props;
    HIP_ERROR_CHECK(hipGetDeviceProperties(&props, device_id));
    throw std::runtime_error("Unsupported device architecture - ID: " + std::to_string(device_id)
                             + ", Normalized Name: " + device_name + ", Raw Name: " + props.gcnArchName);
}

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
bool IsGpuPointer(const void* ptr)
{
    hipPointerAttribute_t attr;
    HIP_ERROR_CHECK(hipPointerGetAttributes(&attr, ptr));

    return attr.type == hipMemoryTypeDevice;
}

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