#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_bias_permute_op.h"

namespace flashck {

template<typename T>
GemmRCRBiasPermuteOp<T>::GemmRCRBiasPermuteOp(Shape permute_shape, std::string permute_layout):
    GemmRCRBiasOp<T>::GemmRCRBiasOp("gemm_rcr_bias_permute_op")
{
    this->op_name_       = "gemm_rcr_bias_permute_" + permute_layout;
    this->op_kind_       = GemmOperationKind::GemmPermuteM2N3;
    this->permute_shape_ = permute_shape;
}

template<typename T>
Shape GemmRCRBiasPermuteOp<T>::InferTrueShape(Variable* a, Variable* b)
{
    // m2n3
    Shape output_shape = GemmRCROp<T>::InferShapeImpl(a, b);

    DDim batch_size = output_shape.GetDim(0);
    VLOG(1) << "batch_size: " << batch_size.ToString();
    DDim seq_len = output_shape.GetDim(1);
    VLOG(1) << "seq_len: " << seq_len.ToString();
    DDim hidden_units = output_shape.GetDim(2);
    VLOG(1) << "hidden_units: " << hidden_units.ToString();

    this->permute_shape_.InsertDim(0, seq_len);
    auto t1 = this->permute_shape_.GetDim(1);  // 3
    auto t2 = this->permute_shape_.GetDim(2);  // num_heads

    VLOG(1) << "permute_shape: " << this->permute_shape_.ToString();
    // (B, seqlen, dim) * (3*dim, dim) = (B, seqlen, 3*dim)
    // reshape to : (B, seqlen, 3, num_heads, head_dim)
    // output : (3, B, num_heads, seqlen, head_dim) 20314
    // output: (3, B, seqlen, num_heads, head_dim) 20134
    // Shape reshape_output_shape({t1, batch_size, t2, seq_len, hidden_units / t1 / t2});

    Shape reshape_output_shape({t1, batch_size, seq_len, t2, hidden_units / t1 / t2});

    return reshape_output_shape;
}

template<typename T>
Variable* GemmRCRBiasPermuteOp<T>::operator()(Variable* a, Variable* b, Variable* bias)
{
    this->AlignAB(a, b);
    this->input_var_ = {a, b, bias};
    this->SanityCheck(a, b);

    Shape output_shape = InferTrueShape(a, b);

    auto max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    VLOG(1) << "output_shape: " << output_shape.ToString();
    this->output_var_ = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->SetParentsNode({a, b, bias});
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template class GemmRCRBiasPermuteOp<float>;
template class GemmRCRBiasPermuteOp<_Float16>;
template class GemmRCRBiasPermuteOp<ushort>;

}  // namespace flashck