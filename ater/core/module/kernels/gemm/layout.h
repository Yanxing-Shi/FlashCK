#pragma once

#include <string>

#include "ater/core/profiler/gemm_operation.h"
#include "ater/core/profiler/library.h"

/*
Layout class
*/

namespace ater {

class Layout {
    virtual bool CheckOutputLayout(const std::shared_ptr<void>& kernel) = 0;
    virtual bool CheckInputLayout(const std::shared_ptr<void>& kernel)  = 0;
};

class RCRLayout: public Layout {
public:
    std::string ck_layout_a = "ck::tensor_layout::gemm::RowMajor";
    std::string ck_layout_b = "ck::tensor_layout::gemm::ColumnMajor";
    std::string ck_layout_c = "ck::tensor_layout::gemm::RowMajor";
    std::string stride_a    = "K";
    std::string stride_b    = "K";
    std::string stride_c    = "N";

    std::string args_parse = R"(
    int64_t M = std::stoi(argv[1]);
    int64_t N = std::stoi(argv[2]);
    int64_t K = std::stoi(argv[3]);
    int64_t split_k = std::atoi(argv[4]);
    int64_t a_dim0 = M;
    int64_t a_dim1 = K;
    int64_t b_dim0 = N;
    int64_t b_dim1 = K;
    int64_t c_dim0 = M;
    int64_t c_dim1 = N;
)";

    bool CheckOutputLayout(const std::shared_ptr<void>& kernel) override
    {
        auto gemm_kernel = std::static_pointer_cast<GemmOperation>(kernel);
        return gemm_kernel->c_tensor_desc_.layout_ == LayoutType::RowMajor;
    }

    bool CheckInputLayout(const std::shared_ptr<void>& kernel) override
    {
        auto gemm_kernel = std::static_pointer_cast<GemmOperation>(kernel);
        return gemm_kernel->a_tensor_desc_.layout_ == LayoutType::RowMajor
               && gemm_kernel->b_tensor_desc_.layout_ == LayoutType::ColumnMajor;
    }
};

}  // namespace ater