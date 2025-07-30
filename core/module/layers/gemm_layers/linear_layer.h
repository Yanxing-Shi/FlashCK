#pragma once

#include <memory>

#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/graph/node.h"

#include "core/module/operations/gemm_ops/gemm_epilogue_ops/gemm_rcr_bias_add_op.h"
#include "core/module/operations/gemm_ops/gemm_epilogue_ops/gemm_rcr_bias_gelu_op.h"
#include "core/module/operations/gemm_ops/gemm_epilogue_ops/gemm_rcr_bias_multiply_op.h"
#include "core/module/operations/gemm_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"
#include "core/module/operations/gemm_ops/gemm_epilogue_ops/gemm_rcr_bias_silu_op.h"
#include "core/module/operations/gemm_ops/gemm_epilogue_ops/gemm_rcr_bias_tanh_op.h"
#include "core/module/operations/gemm_ops/gemm_epilogue_ops/gemm_rcr_op.h"

namespace flashck {

template<typename T>
class LinearLayer: public Layer {
public:
    LinearLayer(int64_t     in_channels,
                int64_t     out_channels,
                bool        use_bias       = true,
                std::string specialization = "");

    ~LinearLayer() = default;

    Variable* operator()(Variable* a, Variable* d0 = nullptr);

    void LoadParam(const T* weight_ptr, const T* bias_ptr = nullptr);

    std::unique_ptr<Variable> weight_var_;
    std::unique_ptr<Variable> bias_var_;

    std::unique_ptr<GemmRCROp<T>>             gemm_rcr_op_;
    std::unique_ptr<GemmRCRBiasOp<T>>         gemm_rcr_bias_op_;
    std::unique_ptr<GemmRCRBiasGeluOp<T>>     gemm_rcr_bias_gelu_op_;
    std::unique_ptr<GemmRCRBiasSiLUOp<T>>     gemm_rcr_bias_silu_op_;
    std::unique_ptr<GemmRCRBiasTanhOp<T>>     gemm_rcr_bias_tanh_op_;
    std::unique_ptr<GemmRCRBiasMultiplyOp<T>> gemm_rcr_bias_multiply_op_;
    std::unique_ptr<GemmRCRBiasAddOp<T>>      gemm_rcr_bias_add_op_;
    std::unique_ptr<SplitKGemmRCROp<T>>       split_k_gemm_rcr_op_;

    std::string gemm_op_name_;
};
}  // namespace flashck