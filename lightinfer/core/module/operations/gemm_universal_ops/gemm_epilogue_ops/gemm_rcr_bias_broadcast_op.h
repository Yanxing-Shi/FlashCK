#pragma once

#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_op.h"

namespace lightinfer {

/*
gemm_rcr_bias with 2 extra sources.
BinaryOp2(BinaryOp1(UnaryOp(TensorOp(X) + bias), residual1), residual2)
*/

template<typename T>
class GemmRCRBiasBroadcastOp: public GemmRCRBiasOp<T> {
public:
    GemmRCRBiasBroadcastOp(std::string op_name = "gemm_rcr_bias_broadcast_op"): GemmRCRBiasOp<T>(op_name) {};

    using GemmRCRBiasOp<T>::IsVaildInputs;
    void IsVaildInputs(const std::vector<Variable*>& input_var);

    Variable* operator()(Variable* a, Variable* b, Variable* bias, Variable* d0);
};

}  // namespace lightinfer