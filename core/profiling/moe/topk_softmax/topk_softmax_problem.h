#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/gemm/gemm_library.h"
#include "core/utils/dtype.h"

namespace flashck {

/**
 * @class TopKSoftmaxProblem
 * @brief Represents a TopK Softmax operation problem configuration
 *
 */
class TopKSoftmaxProblem: public ProblemBase<TopKSoftmaxProblem> {
public:

    std::string GetTypeImpl()
    {
        return "TopKSoftmaxProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        

        return oss.str();
    }
    // ====================== Problem Configuration ======================
    DataType input_dtype_;
    DataType weight_dtype_;
    DataType index_dtype_ = DataType::INT32;


    int64_t num_tokens_;
    int64_t num_experts_;

    int64_t topk_;

    int64_t input_stride_;
    int64_t output_stride_;
};

}  // namespace flashck