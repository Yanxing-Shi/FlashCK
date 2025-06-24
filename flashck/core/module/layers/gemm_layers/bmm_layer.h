#pragma once

#include <functional>
#include <memory>
#include <string>

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"

#include "flashck/core/module/operations/gemm_universal_ops/bmm_epilogue_ops/bmm_rcr_op.h"

namespace flashck {

template<typename T>
class BmmLayer: public Layer {
public:
    BmmLayer(int64_t     in_channels,
             int64_t     out_channels,
             bool        use_bias       = true,
             std::string specialization = "",
             Shape       permute_shape  = {});

    ~BmmLayer() = default;

    Variable* operator()(Variable* a, Variable* d0 = nullptr);

    // void BeforeForward(DDim seq_len_dim);

    void LoadParam(const T* weight_ptr, const T* bias_ptr = nullptr);

    int64_t in_channels_;
    int64_t out_channels_;
    bool    use_bias_;

    Shape permute_shape_;

    std::unique_ptr<Variable> weight_var_;
    std::unique_ptr<Variable> bias_var_;

    std::unique_ptr<BmmRCROp<T>> bmm_rcr_op_;

    std::string bmm_op_name_;
};
}  // namespace flashck