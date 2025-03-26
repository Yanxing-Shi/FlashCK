#include "lightinfer/core/utils/rocm_info.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/errors.h"
#include "lightinfer/core/utils/flags.h"
#include "lightinfer/core/utils/string_utils.h"

LI_DECLARE_string(selected_gpus);

static std::once_flag                               g_device_props_size_init_flag;
static std::vector<std::unique_ptr<std::once_flag>> g_device_props_init_flags;
static std::vector<hipDeviceProp_t>                 g_device_props;

namespace lightinfer {
// GPU device count
int GetGPUDeviceCount()
{

    const auto* hip_visible_devices = std::getenv("HIP_VISIBLE_DEVICES");
    if (hip_visible_devices != nullptr) {
        std::string hip_visible_devices_str(hip_visible_devices);
        if (!hip_visible_devices_str.empty()) {
            hip_visible_devices_str.erase(0, hip_visible_devices_str.find_first_not_of('\''));
            hip_visible_devices_str.erase(hip_visible_devices_str.find_last_not_of('\'') + 1);
            hip_visible_devices_str.erase(0, hip_visible_devices_str.find_first_not_of('\"'));
            hip_visible_devices_str.erase(hip_visible_devices_str.find_last_not_of('\"') + 1);
        }

        if (std::all_of(
                hip_visible_devices_str.begin(), hip_visible_devices_str.end(), [](char ch) { return ch == ' '; })) {
            VLOG(2) << "HIP_VISIBLE_DEVICES is set to be "
                       "empty. No GPU detected.";
            return 0;
        }
    }

    int device_count = 0;
    LI_ENFORCE_HIP_SUCCESS(hipGetDeviceCount(&device_count));
    return device_count;
}

int GetGPUComputeCapability(int id)
{
    LI_ENFORCE_LT(id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  id,
                                  GetGPUDeviceCount()));
    int  major, minor;
    auto major_error_code = hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, id);
    auto minor_error_code = hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, id);

    LI_ENFORCE_HIP_SUCCESS(major_error_code);
    LI_ENFORCE_HIP_SUCCESS(minor_error_code);
    return major * 100 + minor;
}

int GetGPURuntimeVersion(int id)
{
    LI_ENFORCE_LT(id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  id,
                                  GetGPUDeviceCount()));
    int runtime_version = 0;
    LI_ENFORCE_HIP_SUCCESS(hipRuntimeGetVersion(&runtime_version));
    return runtime_version;
}

int GetGPUDriverVersion(int id)
{
    LI_ENFORCE_LT(id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  id,
                                  GetGPUDeviceCount()));
    int driver_version = 0;
    LI_ENFORCE_HIP_SUCCESS(hipDriverGetVersion(&driver_version));
    return driver_version;
}

int GetGPUMultiProcessors(int id)
{
    LI_ENFORCE_LT(id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  id,
                                  GetGPUDeviceCount()));
    int count;
    LI_ENFORCE_HIP_SUCCESS(hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount, id));
    return count;
}

int GetGPUMaxThreadsPerMultiProcessor(int id)
{
    LI_ENFORCE_LT(id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  id,
                                  GetGPUDeviceCount()));
    int count;
    LI_ENFORCE_HIP_SUCCESS(hipDeviceGetAttribute(&count, hipDeviceAttributeMaxThreadsPerMultiProcessor, id));

    return count;
}

int GetGPUMaxThreadsPerBlock(int id)
{
    LI_ENFORCE_LT(id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  id,
                                  GetGPUDeviceCount()));
    int count;
    LI_ENFORCE_HIP_SUCCESS(hipDeviceGetAttribute(&count, hipDeviceAttributeMaxThreadsPerBlock, id));
    return count;
}

std::array<int, 3> GetGpuMaxGridDimSize(int id)
{
    LI_ENFORCE_LT(id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  id,
                                  GetGPUDeviceCount()));
    std::array<int, 3> ret;
    int                size;
    auto               error_code_x = hipDeviceGetAttribute(&size, hipDeviceAttributeMaxGridDimX, id);
    LI_ENFORCE_HIP_SUCCESS(error_code_x);
    ret[0] = size;

    auto error_code_y = hipDeviceGetAttribute(&size, hipDeviceAttributeMaxGridDimY, id);
    LI_ENFORCE_HIP_SUCCESS(error_code_y);
    ret[1] = size;

    auto error_code_z = hipDeviceGetAttribute(&size, hipDeviceAttributeMaxGridDimZ, id);
    LI_ENFORCE_HIP_SUCCESS(error_code_z);
    ret[2] = size;
    return ret;
}
const hipDeviceProp_t& GetDeviceProperties(int id)
{
    std::call_once(g_device_props_size_init_flag, [&] {
        int gpu_num = 0;
        gpu_num     = GetGPUDeviceCount();
        g_device_props_init_flags.resize(gpu_num);
        g_device_props.resize(gpu_num);
        for (int i = 0; i < gpu_num; ++i) {
            g_device_props_init_flags[i] = std::make_unique<std::once_flag>();
        }
    });

    if (id == -1) {
        id = GetCurrentDeviceId();
    }

    if (id < 0 || id >= static_cast<int>(g_device_props.size())) {
        LI_THROW(
            OutOfRange("The device id {} is out of range [0, {}), where {} is the number of devices on this machine.",
                       id,
                       static_cast<int>(g_device_props.size()),
                       static_cast<int>(g_device_props.size())));
    }

    std::call_once(*(g_device_props_init_flags[id]),
                   [&] { LI_ENFORCE_HIP_SUCCESS(hipGetDeviceProperties(&g_device_props[id], id)); });

    return g_device_props[id];
}

std::vector<int> GetSelectedDevices()
{
    // use user specified GPUs in single-node multi-process mode.
    std::vector<int> devices;
    if (!FLAGS_selected_gpus.empty()) {
        auto devices_str = SplitString(FLAGS_selected_gpus, ",");
        for (auto const& id : devices_str) {
            devices.push_back(std::atoi(id.c_str()));
        }
    }
    else {
        int count = GetGPUDeviceCount();
        for (int i = 0; i < count; ++i) {
            devices.push_back(i);
        }
    }
    return devices;
}

int GetCurrentDeviceId()
{
    int device_id;
    LI_ENFORCE_HIP_SUCCESS(hipGetDevice(&device_id));
    return device_id;
}

void SetDeviceId(int device_id)
{
    LI_ENFORCE_LT(device_id,
                  GetGPUDeviceCount(),
                  InvalidArgument("Device id must be less than GPU count, "
                                  "but received id is: {}. GPU count is: {}.",
                                  device_id,
                                  GetGPUDeviceCount()));
    LI_RETRY_HIP_SUCCESS(hipSetDevice(device_id));
}

void GetSetDevice(int input_device_id, int* output_device_id)
{
    if (output_device_id != nullptr) {
        int curr_device_id = GetCurrentDeviceId();
        LI_ENFORCE_HIP_SUCCESS(hipGetDevice(&curr_device_id));
        if (curr_device_id == input_device_id) {
            *output_device_id = curr_device_id;
        }
        else {
            SetDeviceId(input_device_id);
            *output_device_id = input_device_id;
        }
    }
    else {
        SetDeviceId(input_device_id);
    }
}

std::string GetDeviceName(int device_id)
{
    auto props = GetDeviceProperties(device_id);

    const std::string raw_name(props.gcnArchName);

    // https://github.com/ROCmSoftwarePlatform/MIOpen/blob/8498875aef84878e04c1eabefdf6571514891086/src/target_properties.cpp#L40
    static std::map<std::string, std::string> device_name_map = {
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

    const auto name  = raw_name.substr(0, raw_name.find(':'));  // str.substr(0, npos) returns str.
    auto       match = device_name_map.find(name);
    if (match != device_name_map.end())
        return match->second;
    return name;
}

bool IsXdlSupported(int device_id)
{
    return GetDeviceName(device_id) == "gfx908" || GetDeviceName(device_id) == "gfx90a" || GetDeviceName() == "gfx940"
           || GetDeviceName(device_id) == "gfx941" || GetDeviceName(device_id) == "gfx942";
}

bool IsWmmaSupported(int device_id)
{
    return GetDeviceName(device_id) == "gfx1100" || GetDeviceName(device_id) == "gfx1101";
}

std::string GetDeviceArch(int device_id)
{
    if (IsXdlSupported()) {
        return "xdl";
    }
    else if (IsWmmaSupported()) {
        return "wmma";
    }
    else {
        return "legacy";
    }
}

// GPU device synchronize.
void GpuDeviceSynchronize()
{
    LI_ENFORCE_HIP_SUCCESS(hipDeviceSynchronize());
}

/*----------stream------------------*/

void GpuStreamSync(hipStream_t stream)
{
    LI_ENFORCE_HIP_SUCCESS(hipStreamSynchronize(stream));
}

void GpuDestroyStream(hipStream_t stream)
{
    LI_ENFORCE_HIP_SUCCESS(hipStreamDestroy(stream));
}

bool IsGpuPointer(void* ptr)
{
    hipPointerAttribute_t attr;
    auto                  status = hipPointerGetAttributes(&attr, ptr);
    if (status != hipSuccess)
        return false;
    return attr.type == hipMemoryTypeDevice;
}

// Get GPU memory info.
void GetGpuMemoryInfo(std::string name)
{
    size_t free_bytes, total_bytes;
    LI_ENFORCE_HIP_SUCCESS(hipMemGetInfo(&free_bytes, &total_bytes));
    float free  = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used  = total - free;
    LOG(INFO) << name << " free: " << free << " GB, total: " << total << " GB, used: " << used << " GB";
}

}  // namespace lightinfer