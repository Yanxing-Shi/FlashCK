#pragma once

#include "flashck/core/profiling/profiling_engine.h"

#include "flashck/core/module/kernels/norm_kernels/norm_kernel_call_def.h"

namespace flashck {

#define LOAD_SYMBOL(kernel_func, name_str)                                                                             \
    if (!ProfilingEngine::GetInstance()->GetKernelLibrary()->has_symbol(name_str)) {                                   \
        FC_THROW(Unavailable("Kernel symbol not found {}", name_str));                                                 \
    }                                                                                                                  \
    kernel_func = *(ProfilingEngine::GetInstance()->GetKernelLibrary()->get_function<decltype(kernel_func)>(name_str));

}  // namespace flashck