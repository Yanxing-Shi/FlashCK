#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/rocm_info.h"

namespace lightinfer {

// check inf and nan and zero in tensor cuda kernel
template<typename T>
__global__ void inf_nan_zero_checker(
    const T* tensor, const int64_t elem_cnt, const char* tensor_name, int* nan_num, int* pos_inf, int* neg_inf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elem_cnt) {
        float v = (float)(*(tensor + i));
        if (isnan(v)) {
            atomicAdd(nan_num, int(1));
            printf("nan in tensor %s at index %d\n", tensor_name, i);
        }
        auto is_inf = isinf(v);
        if (is_inf) {
            if (v > 0) {
                atomicAdd(pos_inf, int(1));
                printf("+inf in tensor %s at index %d\n", tensor_name, i);
            }
            else {
                atomicAdd(neg_inf, int(1));
                printf("-inf in tensor %s at index %d\n", tensor_name, i);
            }
        }
    }
}

template<typename T>
inline void
ResultChecker(const T* tensor, const int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream = 0)
{
    int nan_num = 0, pos_inf = 0, neg_inf = 0;
    inf_nan_zero_checker<<<1, 1, 0, stream>>>(tensor, elem_cnt, tensor_name.c_str(), &nan_num, &pos_inf, &neg_inf);

    LI_ENFORCE_EQ(
        nan_num == 0 && pos_inf == 0 && neg_inf == 0,
        true,
        Unavailable(
            "Error: nan or inf in tensor {}, nan: {}, +inf: {}, -inf: {}", tensor_name, nan_num, pos_inf, neg_inf));
}

}  // namespace lightinfer