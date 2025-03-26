#pragma once

#include <string>

#include "lightinfer/core/profiler/gemm_operation.h"
#include "lightinfer/core/profiler/library.h"

/*
Layout class
*/

namespace lightinfer {

class Layout {
    virtual bool CheckOutputLayout(const std::shared_ptr<void>& kernel) = 0;
    virtual bool CheckInputLayout(const std::shared_ptr<void>& kernel)  = 0;
};

class RCRLayout: public Layout {
public:
    std::string GetGemmArgsParse()
    {
        std::string args_parse = R"(
    ck::index_t K = std::stoi(argv[1]);
    ck::index_t M = std::stoi(argv[2]);
    ck::index_t N = std::stoi(argv[3]);
    ck::index_t a_dim0 = M;
    ck::index_t a_dim1 = K;
    ck::index_t b_dim0 = N;
    ck::index_t b_dim1 = K;
    ck::index_t c_dim0 = M;
    ck::index_t c_dim1 = N;
    )";
        return args_parse;
    }

    std::string GetSplitKGemmArgsParse()
    {
        std::string args_parse = R"(
    ck::index_t K = std::stoi(argv[1]);
    ck::index_t M = std::stoi(argv[2]);
    ck::index_t N = std::stoi(argv[3]);
    ck::index_t split_k = std::atoi(argv[4]);
    ck::index_t a_dim0 = M;
    ck::index_t a_dim1 = K;
    ck::index_t b_dim0 = N;
    ck::index_t b_dim1 = K;
    ck::index_t c_dim0 = M;
    ck::index_t c_dim1 = N;
    )";
        return args_parse;
    }

    std::string GetBmmArgsParse()
    {
        std::string args_parse = R"(
        ck::index_t B = std::atoi(argv[1]);
        ck::index_t K = std::atoi(argv[2]);
        ck::index_t M = std::atoi(argv[3]);
        ck::index_t N = std::atoi(argv[4]);

        ck::index_t a_dim0 = B;
        ck::index_t a_dim1 = M;
        ck::index_t a_dim2 = K;
        ck::index_t b_dim0 = B;
        ck::index_t b_dim1 = N;
        ck::index_t b_dim2 = K;
        ck::index_t c_dim0 = B;
        ck::index_t c_dim1 = M;
        ck::index_t c_dim2 = N;
    )";
        return args_parse;
    }

    std::string ck_layout_a = "ck::tensor_layout::gemm::RowMajor";
    std::string ck_layout_b = "ck::tensor_layout::gemm::ColumnMajor";
    std::string ck_layout_c = "ck::tensor_layout::gemm::RowMajor";
    std::string stride_a    = "K";
    std::string stride_b    = "K";
    std::string stride_c    = "N";

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
}  // namespace lightinfer