#include "flashck/core/module/operations/gemm_universal_ops/gemm_epilogue_ops/gemm_rcr_op.h"

#include "flashck/core/utils/debug_utils.h"
#include "flashck/core/utils/enforce.h"

#include "flashck/core/graph/node.h"

namespace flashck {

template<typename T>
GemmRCROp<T>::GemmRCROp(std::string op_name): GemmCommonOp<T, GemmRCROp>::GemmCommonOp(op_name)
{
    this->op_name_     = "gemm_rcr";
    this->op_kind_     = GemmOperationKind::Gemm;
    this->epilogue_op_ = TensorOperation::PassThrough;
    this->layout_      = DataLayout::RCR;
}

template<typename T>
Shape GemmRCROp<T>::InferShapeImpl(Variable* a, Variable* b)
{
    auto a_shape_vec = a->GetShape().ToVector();
    auto b_shape_vec = b->GetShape().ToVector();
    auto m           = SliceVec<DDim>(a_shape_vec, 0, a_shape_vec.size() - 1);
    auto n           = b_shape_vec.front();
    m.push_back(std::move(n));

    return Shape(m);
}

template<typename T>
Variable* GemmRCROp<T>::operator()(Variable* a, Variable* b)
{
    this->AlignAB(a, b);
    this->SanityCheck(a, b);
    this->input_var_   = {a, b};
    Shape output_shape = this->InferShape(a, b);
    VLOG(1) << "output_shape: " << output_shape.ToString();
    auto max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_    = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->SetParentsNode({a, b});
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template<typename T>
std::function<std::vector<std::string>(const std::string&)> GemmRCROp<T>::GenBuildCmd()
{
    auto fbuild_cmd = [&](const std::string& exec_key) {
        std::vector<int64_t>     cmd = this->InverseKeyFunc(exec_key);
        std::vector<std::string> cmd_str;
        std::transform(
            cmd.begin(), cmd.end(), std::back_inserter(cmd_str), [](int64_t i) { return std::to_string(i); });
        return cmd_str;
    };

    return fbuild_cmd;
}

// (M, K) * (N, K) = (M, N)
// profiling always uses 2d * 2d.
template<typename T>
std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> GemmRCROp<T>::ExtractDimsImpl(bool for_profiling)
{
    auto a_shape_size = this->input_var_[0]->GetShape().GetNumDim();

    std::vector<int64_t> dim_idx_0{0};
    std::vector<int64_t> dim_idx_1{1};

    if (for_profiling) {
        return {{"M",
                 {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_0),
                  std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_0)}},
                {"N",
                 {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_0),
                  std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_1)}},
                {"K",
                 {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_1),
                  std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_1)}}};
    }
    else {
        std::vector<int64_t> dim_idx(a_shape_size - 1);
        std::iota(dim_idx.begin(), dim_idx.end(), 0);
        std::vector<int64_t> dim_idx_shape{a_shape_size - 1};
        return {{"M",
                 {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx),
                  std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx)}},
                {"N",
                 {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_0),
                  std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_shape)}},
                {"K",
                 {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_shape),
                  std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_1)}}};
    }
}

template<typename T>
void GemmRCROp<T>::ForwardImpl()
{
    auto a        = this->GetParentNode(0);
    auto b        = this->GetParentNode(1);
    auto bias_ptr = this->input_var_.size() > 2 ? (T*)this->GetParentNode(2)->GetValue() : nullptr;
    auto d0_ptr   = this->num_tpls_ >= 1 ? (T*)this->GetParentNode(3)->GetValue() : nullptr;
    auto c        = this->GetChildNode(0);

    T* in_ptr     = (T*)a->GetValue();
    T* weight_ptr = (T*)b->GetValue();
    T* out_ptr    = (T*)c->GetValue();

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    Shape out_shape = this->InferShape(a, b);
    VLOG(1) << "gemm " << this->op_name_ << ", out shape: " << out_shape.ToString();

    // PrintToScreen(in_ptr, 3, "[" + this->op_name_ + "]" + "in_ptr");
    // PrintToScreen(weight_ptr, 3, "[" + this->op_name_ + "]" + "weight_ptr");
    // PrintToScreen(bias_ptr, 3, "[" + this->op_name_ + "]" + "bias_ptr");
    // PrintToScreen(d0_ptr, 3, "[" + this->op_name_ + "]" + "d0_ptr");

    this->output_var_[0]->SetShape(out_shape);

    auto broadcast_shape_func = [&](Variable* x) {
        auto x_shape_vec = x->GetShape().ToVector();
        int  dim0_value  = 1;
        std::for_each(
            x_shape_vec.begin(), x_shape_vec.end() - 1, [&](const DDim& dim) { dim0_value *= dim.GetValues()[0]; });

        return dim0_value;
    };

    GemmKernelArgs gemm_args;
    gemm_args.in_ptr_     = in_ptr;
    gemm_args.weight_ptr_ = weight_ptr;
    gemm_args.out_ptr_    = out_ptr;
    gemm_args.bias_ptr_   = bias_ptr;
    gemm_args.d0_ptr_     = d0_ptr;
    gemm_args.a_dim0_     = broadcast_shape_func(a);
    gemm_args.a_dim1_     = a->GetShape().GetLastDim().GetValues()[0];
    gemm_args.b_dim0_     = b->GetShape().GetDim(0).GetValues()[0];
    gemm_args.b_dim1_     = b->GetShape().GetDim(1).GetValues()[0];
    gemm_args.c_dim0_     = broadcast_shape_func(c);
    gemm_args.c_dim1_     = c->GetShape().GetLastDim().GetValues()[0];
    gemm_args.p_dim0_     = this->permute_shape_.GetNumDim() ? a->GetShape().GetDim(1).GetValues()[0] : -1;  // sen_dim
    gemm_args.p_dim1_     = this->permute_shape_.GetNumDim() ? this->permute_shape_.GetDim(1).GetValues()[0] : -1;
    gemm_args.p_dim2_     = this->permute_shape_.GetNumDim() ? this->permute_shape_.GetDim(2).GetValues()[0] : -1;
    gemm_args.stream_     = this->context_ptr_->GetStream();

    this->register_kernel_ptr_->KernelLauncher(this->GetName(), std::move(gemm_args));

    // PrintToScreen(out_ptr, 3, "[" + this->op_name_ + "]" + "out_ptr");
    // ResultChecker(out_ptr, std::get<0>(out_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "out_ptr");
}

template class GemmRCROp<float>;
template class GemmRCROp<_Float16>;
template class GemmRCROp<ushort>;

}  // namespace flashck