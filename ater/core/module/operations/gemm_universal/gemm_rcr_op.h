#pragma once

#include "ater/core/module/kernels/gemm/gemm_rcr_kernel.h"
#include "ater/core/module/operations/gemm_universal/gemm_common_op.h"

namespace ater {

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
class GemmRCROp: public GemmCommonOp<T> {
public:
    GemmRCROp();
    ~GemmRCROp() = default;

    Shape InferShape(Variable* A, Variable* B);

    std::vector<int> InvertExecKey(const std::string& key);

    std::vector<std::string> GenProfileCmd(const std::string& profiler_prefix,
                                           const std::string& profiler_filename,
                                           const std::string& exec_key);

    // (M, K) * (N, K) = (M, N)
    // profiling always uses 2d * 2d.
    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDims(bool for_profiling = true);

    void AlignAB(Variable* A, Variable* B);
};

}  // namespace ater