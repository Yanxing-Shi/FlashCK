#pragma once

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>

#include "tests/utils/torch_utils.h"

#include "flashck/core/utils/log.h"
#include "flashck/core/utils/memory_utils.h"

#include "flashck/core/graph/context.h"

#define EPSILON (1e-20)

// using KernelTestTypes = ::testing::Types<float, _Float16>;
// using KernelTestTypes = ::testing::Types<_Float16>;
using KernelTestTypes = ::testing::Types<ushort>;
// using KernelTestTypes = ::testing::Types<float>;

inline bool AlmostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8)
{

    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }

    if (std::isinf(a) || std::isinf(b)) {
        return false;
    }

    if (a == 0 && b == 0)
        return true;

    double scale     = std::max(std::abs(a), std::abs(b));
    double threshold = atol + rtol * scale;
    return (std::abs(a - b) <= threshold);
}

template<typename out_type, typename ref_type>
inline bool CheckResultImpl(const std::string& name, out_type* out, ref_type* ref, size_t size, float atol, float rtol)
{
    size_t failures     = 0;
    float  relative_gap = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        // The values for the output and the reference.
        float a = (float)out[i];
        float b = (float)ref[i];

        bool ok = AlmostEqual(a, b, atol, rtol);
        // Print the error.
        if (!ok && failures < 4) {
            LOG(ERROR) << ">> invalid result for i=" << i << ":";
            LOG(ERROR) << ">>    found......: " << a;
            LOG(ERROR) << ">>    expected...: " << b;
            LOG(ERROR) << ">>    error......: " << std::fabsf(a - b);
            LOG(ERROR) << ">>    tol........: " << atol + rtol * std::fabs(b);
        }
        // Update the number of failures.
        failures += ok ? 0 : 1;
        // Update the relative gap.
        relative_gap += std::fabsf(a - b) / (std::fabsf(b) + EPSILON);
    }

    relative_gap /= size;

    // Allow not matched up to 1% elements.
    size_t tol_failures = (size_t)(0.01 * size);

    std::string flag = failures <= tol_failures ? "....OK" : "FAILED";

    LOG(INFO) << "check..." << flag << " : " << name << " (failures: " << 100. * failures / size << "% atol: " << atol
              << " rtol: " << rtol << " rel_gap: " << 100. * relative_gap << "%)";

    return failures <= tol_failures;
}

template<typename Y, typename X>
inline Y bit_cast(const X& x)
{
    // static_assert(__has_builtin(__builtin_bit_cast), "");
    // static_assert(sizeof(X) == sizeof(Y), "Do not support cast between different size of type");

    return __builtin_bit_cast(Y, x);
}

inline float bf16_to_float_raw(uint16_t x)
{
    union {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(x) << 16};
    return u.fp32;
}

template<typename ref_type>
inline bool CheckResultImpl(const std::string& name, ushort* out, ref_type* ref, size_t size, float atol, float rtol)
{
    size_t failures     = 0;
    float  relative_gap = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        // The values for the output and the reference.
        float a = bf16_to_float_raw(bit_cast<uint16_t>(out[i]));
        float b = (float)ref[i];

        bool ok = AlmostEqual(a, b, atol, rtol);
        // Print the error.
        if (!ok && failures < 4) {
            LOG(ERROR) << ">> invalid result for i=" << i << ":";
            LOG(ERROR) << ">>    found......: " << a;
            LOG(ERROR) << ">>    expected...: " << b;
            LOG(ERROR) << ">>    error......: " << std::fabsf(a - b);
            LOG(ERROR) << ">>    tol........: " << atol + rtol * std::fabs(b);
        }
        // Update the number of failures.
        failures += ok ? 0 : 1;
        // Update the relative gap.
        relative_gap += std::fabsf(a - b) / (std::fabsf(b) + EPSILON);
    }

    relative_gap /= size;

    // Allow not matched up to 1% elements.
    size_t tol_failures = (size_t)(0.01 * size);

    std::string flag = failures <= tol_failures ? "....OK" : "FAILED";

    LOG(INFO) << "check..." << flag << " : " << name << " (failures: " << 100. * failures / size << "% atol: " << atol
              << " rtol: " << rtol << " rel_gap: " << 100. * relative_gap << "%)";

    return failures <= tol_failures;
}

template<typename out_type, typename ref_type>
inline bool CheckResult(
    const std::string& name, out_type* out, ref_type* ref, size_t size, bool device_out = true, bool device_ref = true)
{
    float atol = CppTypeToTorchType<out_type>::atol;
    float rtol = CppTypeToTorchType<out_type>::rtol;

    out_type* h_out = nullptr;
    if (device_out) {
        h_out = reinterpret_cast<out_type*>(malloc(sizeof(out_type) * size));
        flashck::HipD2HCpy(h_out, out, size);
        out = h_out;
    }
    ref_type* h_ref = nullptr;
    if (device_ref) {
        h_ref = reinterpret_cast<ref_type*>(malloc(sizeof(ref_type) * size));
        flashck::HipD2HCpy(h_ref, ref, size);
        ref = h_ref;
    }

    bool is_ok = CheckResultImpl(name, out, ref, size, atol, rtol);
    if (h_out != nullptr) {
        free(h_out);
    }
    if (h_ref != nullptr) {
        free(h_ref);
    }
    return is_ok;
}

class TestBase: public ::testing::Test {
public:
    void SetUp() override
    {
        // hipGetDevice(&dev_id_);
        // hipStreamCreate(&stream_);

        torch::manual_seed(seed_);
        // context_ptr_->SetStream(stream_);
    }

    void TearDown() override
    {
        // hipStreamDestroy(stream_);
    }

protected:
    uint32_t seed_ = 0;

    int                               dev_id_ = 0;
    hipStream_t                       stream_;
    std::shared_ptr<flashck::Context> context_ptr_;
};
