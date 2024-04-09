#pragma once

#include <functional>
#include <string>

#include "ater/core/graph/layer.h"
#include "ater/core/graph/node.h"
#include "ater/core/utils/layout.h"

#include "ater/core/module/operations/gemm_universal/gemm_common_op.h"

namespace ater {

template<typename T>
class LinearLayer: public Layer {
public:
    LinearLayer(int         in_channels,
                int         out_channels,
                DataLayout  layout         = DataLayout::RCR,
                bool        use_bias       = false,
                std::string specialization = "");

    ~LinearLayer();

    Variable* operator()(Variable* a);

    void LoadParam(const T* para_ptr);

private:
    int        in_channels_;
    int        out_channels_;
    DataLayout layout_;
    bool       use_bias_;

    Variable* weight_var_;
    Variable* bias_var_;

    std::shared_ptr<GemmCommonOp<T>> gemm_op_;

    std::string gemm_op_name_ = "Gemm";
};
}  // namespace ater