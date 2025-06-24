#include "flashck/core/module/operations/gemm_universal_ops/bmm_epilogue_ops/bmm_rcr_op.h"

#include "flashck/core/module/kernels/gemm_kernels/bmm_epilogue_kernels/bmm_rcr_kernel.h"

namespace flashck {

template<typename CppType>
BmmRCROp<CppType>::BmmRCROp(std::string op_name): GemmCommonOp<CppType, BmmRCROp>::GemmCommonOp(op_name)
{
    this->op_name_     = "bmm_rcr";
    this->op_kind_     = GemmOperationKind::BatchGemm;
    this->epilogue_op_ = TensorOperation::PassThrough;
    this->layout_      = DataLayout::RCR;
}

template<typename CppType>
Shape BmmRCROp<CppType>::InferShapeImpl(Variable* a, Variable* b)
{
    auto batch_size = this->GetBatchSize(a, b);
    auto m          = a->GetShape().GetDim(a->GetShape().GetNumDim() - 2);
    auto n          = b->GetShape().GetDim(b->GetShape().GetNumDim() - 2);
    return Shape({batch_size, m, n});
}

template<typename CppType>
Variable* BmmRCROp<CppType>::operator()(Variable* a, Variable* b)
{
    this->AlignAB(a, b);
    this->SanityCheck(a, b);
    this->input_var_   = {a, b};
    Shape output_shape = this->InferShape(a, b);
    VLOG(1) << "output_shape: " << output_shape.ToString();
    auto max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_    = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<CppType>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->SetParentsNode({a, b});
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template<typename CppType>
std::function<std::vector<std::string>(const std::string&)> BmmRCROp<CppType>::GenBuildCmd()
{
    auto fbuild_cmd = [&](const std::string& exec_key) {
        std::vector<int64_t>     cmd = this->InverseKeyFunc(exec_key);  // {B, K, N, K, O}
        std::vector<std::string> cmd_str;
        std::transform(
            cmd.begin(), cmd.end(), std::back_inserter(cmd_str), [](int64_t i) { return std::to_string(i); });
        return cmd_str;
    };
    return fbuild_cmd;
}

template<typename CppType>
std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> BmmRCROp<CppType>::ExtractDimsImpl(bool for_profiling)
{
    // C = A * B
    // A shape is(B, M, K) for row - major layout and (B, K, M) for column - major layout
    // B shape is(B, K, N) for row - major layout and (B, N, K) for column - major layout
    // C shape is(B, M, N) for row - major layout and (B, N, M) for column - major layout
    auto a_shape      = this->input_var_[0]->GetShape();
    auto b_shape      = this->input_var_[1]->GetShape();
    auto output_shape = this->output_var_[0]->GetShape();

    auto batch_dim = this->CreateInputBatchDimInfo({a_shape, b_shape}, {0, 0}, output_shape.GetDim(0));
    batch_dim.push_back(std::make_shared<DimInfo>(TensorSource::kOutput, 0, output_shape.GetDim(0).GetValues()));

    std::vector<int64_t> m_in_a = {a_shape.GetNumDim() - 2};
    std::vector<int64_t> m_in_c = {1};
    std::vector<int64_t> n_in_b = {b_shape.GetNumDim() - 2};
    std::vector<int64_t> n_in_c = {2};
    std::vector<int64_t> k_in_a = {a_shape.GetNumDim() - 1};
    std::vector<int64_t> k_in_b = {b_shape.GetNumDim() - 1};

    return {{"B", batch_dim},
            {"M",
             {std::make_shared<DimInfo>(TensorSource::kInput, 0, m_in_a),
              std::make_shared<DimInfo>(TensorSource::kOutput, 0, m_in_c)}},
            {"N",
             {std::make_shared<DimInfo>(TensorSource::kInput, 1, n_in_b),
              std::make_shared<DimInfo>(TensorSource::kOutput, 0, n_in_c)}},
            {"K",
             {std::make_shared<DimInfo>(TensorSource::kInput, 0, k_in_a),
              std::make_shared<DimInfo>(TensorSource::kInput, 1, k_in_b)}}};
}

template<typename CppType>
void BmmRCROp<CppType>::ForwardImpl()
{
    CppType* in_ptr     = (CppType*)this->GetParentNode(0)->GetValue();
    CppType* weight_ptr = (CppType*)this->GetParentNode(1)->GetValue();
    CppType* out_ptr    = (CppType*)this->GetChildNode(0)->GetValue();

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    Shape out_shape = this->InferShape(this->GetParentNode(0), this->GetParentNode(1));
    this->output_var_[0]->SetShape(out_shape);
    int64_t out_dim0_value = out_shape.GetDim(0).GetValues()[0];
    int64_t out_dim1_value = out_shape.GetDim(1).GetValues()[0];
    int64_t out_dim2_value = out_shape.GetDim(2).GetValues()[0];

    GemmKernelArgs gemm_args;
    gemm_args.in_ptr_     = in_ptr;
    gemm_args.weight_ptr_ = weight_ptr;
    gemm_args.out_ptr_    = out_ptr;

    auto a_num_dim = this->GetParentNode(0)->GetShape().GetNumDim();
    auto b_num_dim = this->GetParentNode(1)->GetShape().GetNumDim();

    if (a_num_dim == 3 && b_num_dim == 3) {
        gemm_args.a_dim0_ = this->GetParentNode(0)->GetShape().GetDim(0).GetValues()[0];
        gemm_args.a_dim1_ = this->GetParentNode(0)->GetShape().GetDim(1).GetValues()[0];
        gemm_args.a_dim2_ = this->GetParentNode(0)->GetShape().GetDim(2).GetValues()[0];

        gemm_args.b_dim0_ = this->GetParentNode(1)->GetShape().GetDim(0).GetValues()[0];
        gemm_args.b_dim1_ = this->GetParentNode(1)->GetShape().GetDim(1).GetValues()[0];
        gemm_args.b_dim2_ = this->GetParentNode(1)->GetShape().GetDim(2).GetValues()[0];
    }
    else if (a_num_dim == 2 && b_num_dim == 3) {
        gemm_args.a_dim0_ = 1;
        gemm_args.a_dim1_ = this->GetParentNode(0)->GetShape().GetDim(0).GetValues()[0];
        gemm_args.a_dim2_ = this->GetParentNode(0)->GetShape().GetDim(1).GetValues()[0];

        gemm_args.b_dim0_ = this->GetParentNode(1)->GetShape().GetDim(0).GetValues()[0];
        gemm_args.b_dim1_ = this->GetParentNode(1)->GetShape().GetDim(1).GetValues()[0];
        gemm_args.b_dim2_ = this->GetParentNode(1)->GetShape().GetDim(2).GetValues()[0];
    }
    else if (a_num_dim == 3 && b_num_dim == 2) {
        gemm_args.a_dim0_ = this->GetParentNode(0)->GetShape().GetDim(0).GetValues()[0];
        gemm_args.a_dim1_ = this->GetParentNode(0)->GetShape().GetDim(1).GetValues()[0];
        gemm_args.a_dim2_ = this->GetParentNode(0)->GetShape().GetDim(2).GetValues()[0];

        gemm_args.b_dim0_ = this->GetParentNode(1)->GetShape().GetDim(0).GetValues()[0];
        gemm_args.b_dim1_ = this->GetParentNode(1)->GetShape().GetDim(0).GetValues()[0];
        gemm_args.b_dim2_ = this->GetParentNode(1)->GetShape().GetDim(1).GetValues()[0];
    }
    else {
        LI_THROW(Unavailable("Unsupported input shape"));
    }

    gemm_args.stream_ = this->context_ptr_->GetStream();
    gemm_args.c_dim0_ = out_dim0_value;
    gemm_args.c_dim1_ = out_dim1_value;
    gemm_args.c_dim2_ = out_dim2_value;

    KernelKey kernel_key(SourceType::CK, this->layout_, CppTypeToDataType<CppType>::Type());
    auto      register_kernel_ptr = KernelFactory::Instance().SelectKernel(this->op_name_, std::move(kernel_key));

    register_kernel_ptr->KernelLauncher(this->GetName(), std::move(gemm_args));
}

template class BmmRCROp<float>;
template class BmmRCROp<_Float16>;
template class BmmRCROp<ushort>;

}  // namespace flashck