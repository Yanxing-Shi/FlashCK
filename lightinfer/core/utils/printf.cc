#include "lightinfer/core/utils/printf.h"

#include "lightinfer/core/utils/dtype.h"
#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/rocm_info.h"

namespace lightinfer {

std::string HumanReadableSize(double f_size)
{
    size_t                         i    = 0;
    double                         orig = f_size;
    const std::vector<std::string> units({"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"});
    while (f_size >= 1024) {
        f_size /= 1024;
        i++;
    }
    if (i >= units.size()) {
        return Sprintf("{}B", orig);
    }
    return Sprintf("{}{}", f_size, units[i]);
}

template<typename T>
void PrintToFile(const T* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode)
{
    LI_ENFORCE_HIP_SUCCESS(hipDeviceSynchronize());
    DLOG(INFO) << "[INFO] file" << file << "with" << size;
    std::ofstream out_file(file, open_mode);
    if (out_file) {
        T* tmp = new T[size];
        LI_ENFORCE_HIP_SUCCESS(hipMemcpyAsync(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
        for (int i = 0; i < size; ++i) {
            float val = (float)(tmp[i]);
            out_file << val << std::endl;
        }
        delete[] tmp;
    }
    else {
        LI_THROW(Unavailable("Cannot open file: {}", file));
    }
    LI_ENFORCE_HIP_SUCCESS(hipDeviceSynchronize());
}

template void
PrintToFile(const float* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
template void
PrintToFile(const _Float16* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);

template<typename T>
void PrintAbsMean(const T* buf, uint size, hipStream_t stream, std::string name)
{
    if (buf == nullptr) {
        DLOG(WARNING) << "It is an nullptr, skip!";
        return;
    }
    LI_ENFORCE_HIP_SUCCESS(hipDeviceSynchronize());
    T* h_tmp = new T[size];
    LI_ENFORCE_HIP_SUCCESS(hipMemcpyAsync(h_tmp, buf, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    LI_ENFORCE_HIP_SUCCESS(hipDeviceSynchronize());
    double   sum        = 0.0f;
    uint64_t zero_count = 0;
    float    max_val    = -1e10;
    bool     find_inf   = false;
    for (uint i = 0; i < size; i++) {
        if (std::isinf((float)(h_tmp[i]))) {
            find_inf = true;
            continue;
        }
        sum += abs((double)h_tmp[i]);
        if ((float)h_tmp[i] == 0.0f) {
            zero_count++;
        }
        max_val = max_val > abs(float(h_tmp[i])) ? max_val : abs(float(h_tmp[i]));
    }

    DLOG(INFO) << name << " size: " << size << ", abs mean: " << sum / size << ", abs sum: " << sum
               << ", abs max: " << max_val << ", find inf: " << find_inf << "zero count: " << zero_count;

    std::cout << std::endl;
    delete[] h_tmp;
    LI_ENFORCE_HIP_SUCCESS(hipDeviceSynchronize());
}

template void PrintAbsMean(const float* buf, uint size, hipStream_t stream, std::string name);
template void PrintAbsMean(const _Float16* buf, uint size, hipStream_t stream, std::string name);

template<typename T>
void PrintToScreen(const T* result, const int size, const std::string& name)
{
    LOG(INFO) << "------------------------------------------";

    if (result == nullptr) {
        LOG(WARNING) << "name: " << name << ", " << "value: " << "It is an nullptr, skip!";
        return;
    }

    T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * size));
    LI_ENFORCE_HIP_SUCCESS(hipMemcpy(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        LOG(INFO) << "name: " << name << ", " << "index: " << i << ", " << "value: " << static_cast<float>(tmp[i]);
    }

    free(tmp);
}

template<>
void PrintToScreen(const ushort* result, const int size, const std::string& name)
{
    LOG(INFO) << "------------------------------------------";

    if (result == nullptr) {
        LOG(WARNING) << "name: " << name << ", " << "value: " << "It is an nullptr, skip!";
        return;
    }

    ushort* tmp = reinterpret_cast<ushort*>(malloc(sizeof(ushort) * size));
    LI_ENFORCE_HIP_SUCCESS(hipMemcpy(tmp, result, sizeof(ushort) * size, hipMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        LOG(INFO) << "name: " << name << ", " << "index: " << i << ", "
                  << "value: " << bf16_to_float_raw(bit_cast<uint16_t>(tmp[i]));
    }

    free(tmp);
}

template void PrintToScreen(const float* result, const int size, const std::string& name);
template void PrintToScreen(const _Float16* result, const int size, const std::string& name);
template void PrintToScreen(const int32_t* result, const int size, const std::string& name);
template void PrintToScreen(const int64_t* result, const int size, const std::string& name);

template<typename T>
void PrintMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr)
{
    T* tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        LI_ENFORCE_HIP_SUCCESS(hipMemcpy(tmp, ptr, sizeof(T) * m * stride, hipMemcpyDeviceToHost));
        LI_ENFORCE_HIP_SUCCESS(hipDeviceSynchronize());
    }
    else {
        tmp = ptr;
    }

    std::ostringstream os;
    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            os << ii;
        }
        else {
            os << " ";
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                os << static_cast<float>(tmp[ii * stride + jj]);
            }
            else {
                os << jj;
            }
        }
        os << "\n";
    }
    LOG(INFO) << os.str();

    if (is_device_ptr) {
        free(tmp);
    }
}

template void PrintMatrix(float* ptr, int m, int k, int stride, bool is_device_ptr);
template void PrintMatrix(_Float16* ptr, int m, int k, int stride, bool is_device_ptr);

template<typename T>
void CheckMaxVal(const T* result, const int size)
{
    T* tmp = new T[size];
    LI_ENFORCE_HIP_SUCCESS(hipMemcpy(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost));
    float max_val = -100000;
    for (int i = 0; i < size; i++) {
        float val = static_cast<float>(tmp[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    delete[] tmp;
    DLOG(INFO) << "[INFO][HIP] addr " << result << " max val: " << max_val;
}

template void CheckMaxVal(const float* result, const int size);
template void CheckMaxVal(const _Float16* result, const int size);

template<typename T>
void CheckAbsMeanVal(const T* result, const int size)
{
    T* tmp = new T[size];
    LI_ENFORCE_HIP_SUCCESS(hipMemcpy(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += abs(static_cast<float>(tmp[i]));
    }
    delete[] tmp;
    DLOG(INFO) << "[INFO][HIP] addr " << result << " abs mean val: " << sum / size;
}

template void CheckAbsMeanVal(const float* result, const int size);
template void CheckAbsMeanVal(const _Float16* result, const int size);
}  // namespace lightinfer