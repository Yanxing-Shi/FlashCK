#pragma once

#include "flashck/core/module/kernels/gemm_kernels/gemm_epilogue_kernels/gemm_rcr_kernel.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_common_op.h"

#include "flashck/core/graph/node.h"

namespace flashck {

/*
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(N, K).cuda().half()

        y = torch.nn.functional.linear(A, B)
*/
template<typename T>
class GemmRCROp: public GemmCommonOp<T, GemmRCROp<T>> {
public:
    GemmRCROp(std::string op_name = "gemm_rcr");

    ~GemmRCROp() = default;

    Shape InferShapeImpl(Variable* a, Variable* b);

    Variable* operator()(Variable* a, Variable* b);

    std::function<std::vector<std::string>(const std::string&)> GenBuildCmd();

    // (M, K) * (N, K) = (M, N)
    // profiling always uses 2d * 2d.
    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDimsImpl(bool for_profiling);

    void ForwardImpl();
};

}  // namespace flashck