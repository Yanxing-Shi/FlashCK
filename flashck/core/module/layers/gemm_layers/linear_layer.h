#pragma once

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"

#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_add_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_gelu_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_mutiply_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_permute_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_silu_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_tanh_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_op.h"
#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/split_k_gemm_rcr_op.h"

namespace flashck {

template<typename T>
class LinearLayer: public Layer {
public:
    LinearLayer(int64_t     in_channels,
                int64_t     out_channels,
                bool        is_split_k     = false,
                bool        use_bias       = true,
                std::string specialization = "",
                Shape       permute_shape  = {});

    ~LinearLayer() = default;

    Variable* operator()(Variable* a, Variable* d0 = nullptr);

    void LoadParam(const T* weight_ptr, const T* bias_ptr = nullptr);

    int64_t in_channels_;
    int64_t out_channels_;
    bool    use_bias_;

    Shape permute_shape_;

    std::unique_ptr<Variable> weight_var_;
    std::unique_ptr<Variable> bias_var_;

    std::unique_ptr<GemmRCROp<T>>             gemm_rcr_op_;
    std::unique_ptr<GemmRCRBiasOp<T>>         gemm_rcr_bias_op_;
    std::unique_ptr<GemmRCRBiasGeluOp<T>>     gemm_rcr_bias_gelu_op_;
    std::unique_ptr<GemmRCRBiasSiLUOp<T>>     gemm_rcr_bias_silu_op_;
    std::unique_ptr<GemmRCRBiasTanhOp<T>>     gemm_rcr_bias_tanh_op_;
    std::unique_ptr<GemmRCRBiasMultiplyOp<T>> gemm_rcr_bias_multiply_op_;
    std::unique_ptr<GemmRCRBiasAddOp<T>>      gemm_rcr_bias_add_op_;
    std::unique_ptr<GemmRCRBiasPermuteOp<T>>  gemm_rcr_bias_permute_op_;
    std::unique_ptr<SplitKGemmRCROp<T>>       split_k_gemm_rcr_op_;

    std::string gemm_op_name_;
};
}  // namespace flashck