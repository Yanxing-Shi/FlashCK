#pragma once

#include <cstdio>
#include <stdexcept>

#include <glog/logging.h>
#include <hip/hip_runtime.h>
#include <jinja2cpp/template.h>
#include <sqlite3.h>

namespace flashck {

std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params);

std::string GetHipErrorMessage(hipError_t err, const char* call);

#define HIP_ERROR_CHECK(call)                                                                                          \
    do {                                                                                                               \
        const hipError_t _hip_err_ = (call);                                                                           \
        if (_hip_err_ != hipSuccess) [[unlikely]] {                                                                    \
            const std::string _msg_ = GetHipErrorMessage(_hip_err_, #call);                                            \
            LOG(ERROR) << _msg_;                                                                                       \
            throw std::runtime_error(_msg_);                                                                           \
        }                                                                                                              \
    } while (0)

#define HIP_WARN_CHECK(call)                                                                                           \
    do {                                                                                                               \
        const hipError_t _hip_stat_ = (call);                                                                          \
        if (_hip_stat_ != hipSuccess) [[unlikely]] {                                                                   \
            LOG(WARNING) << GetHipErrorMessage(_hip_stat_, #call);                                                     \
        }                                                                                                              \
    } while (0)

}  // namespace flashck
