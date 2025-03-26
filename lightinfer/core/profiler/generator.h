#pragma once

#include <memory>
#include <vector>

#include "lightinfer/core/profiler/embedding_operation.h"
#include "lightinfer/core/profiler/emitters.h"
#include "lightinfer/core/profiler/fmha_fwd_appendkv_operation.h"
#include "lightinfer/core/profiler/fmha_fwd_operation.h"
#include "lightinfer/core/profiler/fmha_fwd_splitkv_operation.h"
#include "lightinfer/core/profiler/gemm_operation.h"
#include "lightinfer/core/profiler/norm_operation.h"
#include "lightinfer/core/profiler/target.h"

#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

class Target;

void CreateGemmOperations(const GemmProblem& problem);

void CreateGemmPermuteOperations(const GemmProblem& problem);

void CreateBmmSoftmaxBmmPermuteOperations(const GemmProblem& problem);

void CreateNormOperations(const NormProblem& problem);

void CreateEmbeddingOperations(const EmbeddingProblem& problem);

void CreateFmhaOperations(const FmhaProblem& problem);

void GenerateXldopsCKKernel(const GenOperationKind&                                                      gen_op_kind,
                            const std::variant<GemmProblem, NormProblem, EmbeddingProblem, FmhaProblem>& problem);

void GenerateCKKernelDispatch(const GenOperationKind&                                                      gen_op_kind,
                              const std::variant<GemmProblem, NormProblem, EmbeddingProblem, FmhaProblem>& problem);

}  // namespace lightinfer