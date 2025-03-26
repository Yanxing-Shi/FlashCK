#include "lightinfer/core/module/layers/gemm_layers/bmm_layer.h"

#include "lightinfer/core/utils/dtype.h"

#include "lightinfer/core/module/operations/gemm_universal_ops/bmm_epilogue_ops/bmm_rcr_op.h"

namespace lightinfer {

template<typename T>
BmmLayer<T>::BmmLayer(
    int64_t in_channels, int64_t out_channels, bool use_bias, std::string specialization, Shape permute_shape):
    Layer("BmmLayer"),
    in_channels_(in_channels),
    out_channels_(out_channels),
    use_bias_(use_bias),
    permute_shape_(permute_shape)
{
    // op
    bmm_op_name_ = use_bias_ ? "bmm_rcr_bias" : "bmm_rcr";

    if (!specialization.empty()) {
        bmm_op_name_ += "_" + specialization;
    }

    if (bmm_op_name_ == "bmm_rcr") {
        bmm_rcr_op_ = std::make_unique<BmmRCROp<T>>();
    }

    // param node
    weight_var_ = std::make_unique<Variable>("weight_var", CppTypeToDataType<T>::Type());
    weight_var_->SetShape({out_channels_, in_channels_});

    if (use_bias) {
        bias_var_ = std::make_unique<Variable>("bias_var", CppTypeToDataType<T>::Type());
        bias_var_->SetShape({out_channels_});
    }
    else {
        bias_var_ = nullptr;
    }

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* BmmLayer<T>::operator()(Variable* a, Variable* d0)
{
    Variable* c = nullptr;

    if (d0 == nullptr) {
        SetInputs({a});
    }
    else {
        SetInputs({a, d0});
    }

    if (bmm_op_name_ == "bmm_rcr") {
        c = (*bmm_rcr_op_)(a, weight_var_.get());
    }

    SetOutputs({c});
    return c;
}

// template<typename T>
// void BmmLayer<T>::BeforeForward(DDim seq_len_dim)
// {
//     if (bmm_op_name_ == "gemm_rcr_bias_permute") {
//         gemm_rcr_op_->UpdateShape(seq_len_dim);
//     }
// }

template<typename T>
void BmmLayer<T>::LoadParam(const T* weight_ptr, const T* bias_ptr)
{
    weight_var_->SetValue((char*)weight_ptr);

    if (use_bias_) {
        bias_var_->SetValue((char*)bias_ptr);
    }
}

template class BmmLayer<float>;
template class BmmLayer<_Float16>;

}  // namespace lightinfer