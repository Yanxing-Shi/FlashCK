#include "ater/core/module/layers/linear_layer.h"

// #include "ater/core/module/operations/gemm_universal/op_map.h"
#include "ater/core/graph/shape.h"
#include "ater/core/module/operations/gemm_universal/gemm_rcr_op.h"
#include "ater/core/utils/dtype.h"

namespace ater {

template<typename T>
LinearLayer<T>::LinearLayer(
    int in_channels, int out_channels, DataLayout layout, bool use_bias, std::string specialization):
    Layer("LinearLayer"), in_channels_(in_channels), out_channels_(out_channels), layout_(layout), use_bias_(use_bias)
{
    if (layout == DataLayout::RCR) {
        gemm_op_name_ = "gemm_rcr";
    }
    else {
        gemm_op_name_ = "gemm_rrr";
    }

    if (use_bias) {
        gemm_op_name_ += "_bias";
    }

    if (!specialization.empty()) {
        gemm_op_name_ += "_" + specialization;
    }

    // auto op_func_ = g_ater_op_map[gemm_op_name_];

    // param node
    weight_var_ = new Variable("weight_var", CppTypeToDataType<T>::Type());
    if (use_bias) {
        bias_var_ = new Variable("bias_var", CppTypeToDataType<T>::Type());
    }
    else {
        bias_var_ = nullptr;
    }

    this->context_ptr_->ExitLayer();  // necessary
}

template<typename T>
LinearLayer<T>::~LinearLayer()
{
}

template<typename T>
Variable* LinearLayer<T>::operator()(Variable* a)
{
    SetInputs({a});
    Variable* c = nullptr;

    if (gemm_op_name_ == "gemm_rcr") {
        gemm_op_ = std::make_shared<GemmRCROp<T>>();
        c        = (*gemm_op_)(a, weight_var_);
    }
    SetOutputs({c});
    return c;
}

template<typename T>
void LinearLayer<T>::LoadParam(const T* para_ptr)
{
    weight_var_->SetValue((char*)para_ptr);
    weight_var_->SetShape(Shape({out_channels_, in_channels_}));
}

template class LinearLayer<float>;
template class LinearLayer<_Float16>;

}  // namespace ater