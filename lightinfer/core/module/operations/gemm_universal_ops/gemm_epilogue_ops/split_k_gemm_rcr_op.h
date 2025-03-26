#pragma once

#include "lightinfer/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/split_k_gemm_rcr_kernel.h"
#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_op.h"

#include "lightinfer/core/graph/node.h"

namespace lightinfer {

template<typename T>
class SplitKGemmRCROp: public GemmRCROp<T> {
public:
    SplitKGemmRCROp(std::string op_name = "split_k_gemm_rcr");

    ~SplitKGemmRCROp() = default;
};

}  // namespace lightinfer