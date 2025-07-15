#pragma once

#include "flashck/core/module/kernels/norm_kernels/norm_kernel_call_def.h"
#include "flashck/core/profiling/profiling_engine.h"

namespace flashck {

// Macro for loading kernel symbols
#define LOAD_SYMBOL(kernel_func, name_str)                                                                             \
    do {                                                                                                               \
        if (!ProfilingEngine::GetInstance()->GetKernelLibrary()->has_symbol(name_str)) {                               \
            FC_THROW(Unavailable("Kernel symbol not found {}", name_str));                                             \
        }                                                                                                              \
        kernel_func = ProfilingEngine::GetInstance()->GetKernelLibrary()->get2_function<decltype(kernel_func)>(        \
            kernel_func_name);                                                                                         \
    } while (0)

}  // namespace flashck