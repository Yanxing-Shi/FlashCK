#pragma once

#include <memory>
#include <vector>

#include "ater/core/profiler/emitters.h"
#include "ater/core/profiler/gemm_operation.h"
#include "ater/core/profiler/target.h"
#include "ater/core/utils/dtype.h"

namespace ater {

class Target;

void CreateGemmOperations(const GemmProblem& problem);

// void GenerateWmmaCKernel(std::shared_ptr<Emitters>& emitter_ptr);

void GenerateXldopsCKKernel(const GenOperationKind& gen_op_kind, const GemmProblem& gemm_problem);

void GenerateCKKernelDispatch(const GenOperationKind& gen_op_kind, const GemmProblem& gemm_problem);

}  // namespace ater