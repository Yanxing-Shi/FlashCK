#include "lightinfer/core/module/layers/gemm_layers/linear_layer.h"

#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

template<typename T>
LinearLayer<T>::LinearLayer(int64_t     in_channels,
                            int64_t     out_channels,
                            bool        is_split_k,
                            bool        use_bias,
                            std::string specialization,
                            Shape       permute_shape):
    Layer("LinearLayer"),
    in_channels_(in_channels),
    out_channels_(out_channels),
    use_bias_(use_bias),
    permute_shape_(permute_shape)
{
    // op
    gemm_op_name_ = use_bias_ ? "gemm_rcr_bias" : "gemm_rcr";

    if (!specialization.empty()) {
        gemm_op_name_ += "_" + specialization;
    }

    if (is_split_k) {
        gemm_op_name_ = "split_k_" + gemm_op_name_;
    }

    if (gemm_op_name_ == "gemm_rcr") {
        gemm_rcr_op_ = std::make_unique<GemmRCROp<T>>();
    }
    else if (gemm_op_name_ == "gemm_rcr_bias") {
        gemm_rcr_bias_op_ = std::make_unique<GemmRCRBiasOp<T>>();
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_silu") {
        gemm_rcr_bias_silu_op_ = std::make_unique<GemmRCRBiasSiLUOp<T>>();
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_gelu") {
        gemm_rcr_bias_gelu_op_ = std::make_unique<GemmRCRBiasGeluOp<T>>();
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_tanh") {
        gemm_rcr_bias_tanh_op_ = std::make_unique<GemmRCRBiasTanhOp<T>>();
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_add") {
        gemm_rcr_bias_add_op_ = std::make_unique<GemmRCRBiasAddOp<T>>();
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_multiply") {
        gemm_rcr_bias_multiply_op_ = std::make_unique<GemmRCRBiasMultiplyOp<T>>();
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_permute") {
        gemm_rcr_bias_permute_op_ = std::make_unique<GemmRCRBiasPermuteOp<T>>(permute_shape_);
    }
    else if (gemm_op_name_ == "split_k_gemm_rcr") {
        split_k_gemm_rcr_op_ = std::make_unique<SplitKGemmRCROp<T>>();
    }
    else {
        LI_THROW(Unavailable("unsupported gemm op {}", gemm_op_name_));
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
Variable* LinearLayer<T>::operator()(Variable* a, Variable* d0)
{
    Variable* c = nullptr;

    if (d0 == nullptr) {
        SetInputs({a});
    }
    else {
        SetInputs({a, d0});
    }

    if (gemm_op_name_ == "gemm_rcr") {
        c = (*gemm_rcr_op_)(a, weight_var_.get());
    }
    else if (gemm_op_name_ == "gemm_rcr_bias") {
        c = (*gemm_rcr_bias_op_)(a, weight_var_.get(), bias_var_.get());
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_silu") {
        c = (*gemm_rcr_bias_silu_op_)(a, weight_var_.get(), bias_var_.get());
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_gelu") {
        c = (*gemm_rcr_bias_gelu_op_)(a, weight_var_.get(), bias_var_.get());
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_tanh") {
        c = (*gemm_rcr_bias_tanh_op_)(a, weight_var_.get(), bias_var_.get());
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_add") {
        c = (*gemm_rcr_bias_add_op_)(a, weight_var_.get(), bias_var_.get(), d0);
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_multiply") {
        c = (*gemm_rcr_bias_multiply_op_)(a, weight_var_.get(), bias_var_.get(), d0);
    }
    else if (gemm_op_name_ == "gemm_rcr_bias_permute") {
        c = (*gemm_rcr_bias_permute_op_)(a, weight_var_.get(), bias_var_.get());
    }
    else if (gemm_op_name_ == "split_k_gemm_rcr") {
        c = (*split_k_gemm_rcr_op_)(a, weight_var_.get());
    }
    else {
        LI_THROW(Unavailable("unsupported gemm op {}", gemm_op_name_));
    }

    SetOutputs({c});
    return c;
}

template<typename T>
void LinearLayer<T>::LoadParam(const T* weight_ptr, const T* bias_ptr)
{
    weight_var_->SetValue((char*)weight_ptr);

    if (use_bias_) {
        bias_var_->SetValue((char*)bias_ptr);
    }
}

template class LinearLayer<float>;
template class LinearLayer<_Float16>;
template class LinearLayer<ushort>;

}  // namespace lightinfer