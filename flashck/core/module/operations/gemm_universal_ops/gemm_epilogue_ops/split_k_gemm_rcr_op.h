#pragma once

#include "flashck/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/split_k_gemm_rcr_kernel.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_op.h"

#include "flashck/core/graph/node.h"

namespace flashck {

template<typename T>
class SplitKGemmRCROp: public GemmRCROp<T> {
public:
    SplitKGemmRCROp(std::string op_name = "split_k_gemm_rcr");

    ~SplitKGemmRCROp() = default;
};

}  // namespace flashck