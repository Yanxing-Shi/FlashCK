#include "flashck/core/module/operations/fmha_ops/fmha_fwd_splitkv_combine_op.h"

#include "flashck/core/module/kernels/fmha_kernels/fmha_fwd_splitkv_combine_kernel.h"

#include "flashck/core/graph/node.h"

namespace flashck {

template<typename T>
FmhaFwdSplitKVCombineOp<T>::FmhaFwdSplitKVCombineOp(
    std::string op_name, FmhaOperationMode op_mode, int64_t q_num_heads, int64_t v_head_dim, int64_t num_splits):
    FmhaCommonOp<T, FmhaFwdSplitKVCombineOp<T>>::FmhaCommonOp(op_name)
{
    FC_ENFORCE_LE(num_splits, 128, Unavailable("num_splits greater than 128 is not supported"));

    this->op_kind_     = FmhaOperationKind::FwdSplitKVCombine;
    this->op_name_     = op_name;
    this->op_mode_     = op_mode;
    this->q_num_heads_ = q_num_heads;
    this->v_head_dim_  = v_head_dim;

    this->num_splits_ = num_splits;
}

template<typename T>
FmhaProblem FmhaFwdSplitKVCombineOp<T>::DefineProblemImpl(const std::vector<int64_t>& inverse_res)
{
    FmhaProblem fmha_problem{CppTypeToDataType<T>::Type(),
                             this->op_mode_,
                             this->op_kind_,
                             this->mask_enum_,
                             this->window_size_,
                             this->bias_enum_,
                             false,  // static quant
                             inverse_res[0],
                             inverse_res[3],
                             inverse_res[3],
                             -1,
                             inverse_res[2],
                             -1,
                             -1,
                             inverse_res[1],
                             this->paged_block_size_,
                             this->rope_enum_,
                             this->rotary_dim_,
                             this->num_splits_};

    return fmha_problem;
}

template<typename T>
std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> FmhaFwdSplitKVCombineOp<T>::ExtractDimsImpl()
{
    std::vector<int64_t> dim_idx_0{0};
    std::vector<int64_t> dim_idx_1{1};
    std::vector<int64_t> dim_idx_2{2};
    std::vector<int64_t> dim_idx_3{3};
    std::vector<int64_t> dim_idx_4{4};

    return {
        // {"num_splits",
        //  {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_0),
        //   std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_0)}},
        {"batch",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_1)}},
        {"seqlen_q",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_1)}},
        // {"seqlen_k",
        //  {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_1),
        //   std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_1)}},
        {"nhead_q",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_2)}},
        // {"nhead_k",
        //  {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_2),
        //   std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_2)}},
        // {"hdim_q",
        //  {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_3),
        //   std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_3)}},
        {"hdim_v",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_4),
          std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_3)}},
    };
}

template<typename T>
Shape FmhaFwdSplitKVCombineOp<T>::InferShape(Variable* out_acc, Variable* lse_acc)
{
    DDim batch_size = lse_acc->GetShape().GetDim(1);
    DDim seq_len    = lse_acc->GetShape().GetDim(2);
    DDim num_heads  = lse_acc->GetShape().GetDim(3);
    DDim head_dim   = out_acc->GetShape().GetDim(4);

    return Shape({batch_size, seq_len, num_heads, head_dim});
}

template<typename T>
void FmhaFwdSplitKVCombineOp<T>::SanityCheck(Variable* out_acc, Variable* lse_acc, Variable* seqstart_q)
{
    // op mode
    if (this->op_mode_ == FmhaOperationMode::Group && lse_acc->GetShape().GetDim(1) != DDim(1)
        && out_acc->GetShape().GetDim(1) != DDim(1)) {
        FC_THROW(Unimplemented("group mode batch size must 1"));
    }

    // num of dimensions
    if (lse_acc->GetShape().GetNumDim() != 4 && out_acc->GetShape().GetNumDim() != 5) {
        FC_THROW(Unimplemented("lse_acc and out acc must be 4D, but got {} and {}",
                               lse_acc->GetShape().GetNumDim(),
                               out_acc->GetShape().GetNumDim()));
    }

    // num splits
    if (lse_acc->GetShape().GetDim(0) != out_acc->GetShape().GetDim(0)) {
        FC_THROW(Unimplemented("lse_acc and out_acc num_splits must be the same, but got {} and {}",
                               lse_acc->GetShape().GetDim(0).ToString(),
                               out_acc->GetShape().GetDim(0).ToString()));
    }

    // batch_size
    if (lse_acc->GetShape().GetDim(1) != out_acc->GetShape().GetDim(1)) {
        FC_THROW(Unimplemented("lse_acc and out_acc batch size must be the same, but got {} and {}",
                               lse_acc->GetShape().GetDim(1).ToString(),
                               out_acc->GetShape().GetDim(1).ToString()));
    }

    // q_num_heads
    if (lse_acc->GetShape().GetDim(3) != out_acc->GetShape().GetDim(3)) {
        FC_THROW(Unimplemented("lse_acc and out_acc q_num_heads must be the same, but got {} and {}",
                               lse_acc->GetShape().GetDim(3).ToString(),
                               out_acc->GetShape().GetDim(3).ToString()));
    }

    // q_seq_len
    if (lse_acc->GetShape().GetDim(2) != out_acc->GetShape().GetDim(2)) {
        FC_THROW(Unimplemented("lse_acc and out_acc q_seq_len must be the same, but got {} and {}",
                               lse_acc->GetShape().GetDim(2).ToString(),
                               out_acc->GetShape().GetDim(2).ToString()));
    }

    // seqstart_q
    if (seqstart_q != nullptr) {
        if (seqstart_q->GetShape().GetNumDim() != 1) {
            FC_THROW(Unimplemented("seqstart_q must be 1D, but got {}", seqstart_q->GetShape().GetNumDim()));
        }

        if (seqstart_q->GetShape().GetDim(0) != lse_acc->GetShape().GetDim(1) + DDim(1)) {
            FC_THROW(Unimplemented("seqstart_q must be batch_size + 1"));
        }

        // if (seqstart_q->GetDtype() != DataType::INT32) {
        //     FC_THROW(Unimplemented("seqstart_q must be int32 tensor, but got seqstart_q: {}",
        //                              DataTypeToString(seqstart_q->GetDtype())));
        // }

        if (this->op_mode_ != FmhaOperationMode::Group) {
            FC_THROW(Unimplemented("seqstart_q and seqstart_k are only used in group mode"));
        }
    }
}

// batch mode
template<typename T>
Variable* FmhaFwdSplitKVCombineOp<T>::operator()(Variable* out_acc, Variable* lse_acc, Variable* seqstart_q)
{
    SanityCheck(out_acc, lse_acc, seqstart_q);

    Shape output_shape    = InferShape(out_acc, lse_acc);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_     = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->input_var_ = {out_acc, lse_acc, seqstart_q};
    std::vector<Node*> parents_node;
    for (auto var : this->input_var_) {
        if (var != nullptr) {
            parents_node.push_back(var);
        }
    }
    this->SetParentsNode(parents_node);
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template<typename T>
std::function<std::vector<std::string>(const std::string&)> FmhaFwdSplitKVCombineOp<T>::GenBuildCmd()
{
    auto fbuild_cmd = [&](const std::string& exec_key) {
        std::vector<int64_t> input_shape = this->ExtractWorkLoad(exec_key);
        // [batch, nhead_q, hdim_v, seqlen_q]
        std::vector<std::string> cmd_str = {"-b=" + std::to_string(input_shape[0]),
                                            "-s=" + std::to_string(input_shape[3]),
                                            "-h=" + std::to_string(input_shape[1]),
                                            "-d_v=" + std::to_string(input_shape[2])};

        return cmd_str;
    };

    return fbuild_cmd;
}

template<typename T>
void FmhaFwdSplitKVCombineOp<T>::ForwardImpl()
{
    auto out_acc = this->GetParentNode(0);
    auto out     = this->GetChildNode(0);

    Variable* lse_acc     = nullptr;
    T*        lse_acc_ptr = nullptr;
    lse_acc               = this->GetParentNode(1);
    lse_acc_ptr           = (T*)lse_acc->GetValue();

    T* out_acc_ptr = (T*)out_acc->GetValue();
    T* out_ptr     = (T*)out->GetValue();

    int64_t* seqstart_q_ptr = nullptr;
    if (this->op_mode_ == FmhaOperationMode::Group) {
        seqstart_q_ptr = (int64_t*)this->GetParentNode(2)->GetValue();
    }

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    PrintToScreen(lse_acc_ptr, 3, "[" + this->op_name_ + "]" + "lse_acc_ptr");
    PrintToScreen(out_acc_ptr, 3, "[" + this->op_name_ + "]" + "out_acc_ptr");
    PrintToScreen(out_ptr, 3, "[" + this->op_name_ + "]" + "out_ptr");
    PrintToScreen(seqstart_q_ptr, 3, "[" + this->op_name_ + "]" + "seqstart_q_ptr");

    Shape out_shape = this->InferShape(lse_acc, out_acc);
    this->output_var_[0]->SetShape(out_shape);

    VLOG(1) << "fmha forward splitkv combine: " << this->op_name_ << " out_shape: " << out_shape.ToString();

    FmhaFwdSplitKVCombineKernelArgs fmha_fwd_splitkv_combine_kernel_args;
    fmha_fwd_splitkv_combine_kernel_args.lse_acc_ptr_ = lse_acc_ptr;
    fmha_fwd_splitkv_combine_kernel_args.out_acc_ptr_ = out_acc_ptr;
    fmha_fwd_splitkv_combine_kernel_args.out_ptr_     = out_ptr;

    fmha_fwd_splitkv_combine_kernel_args.seqstart_q_ptr_ = seqstart_q_ptr;

    fmha_fwd_splitkv_combine_kernel_args.batch_    = out_acc->GetShape().GetDim(1).GetValues()[0];
    fmha_fwd_splitkv_combine_kernel_args.seqlen_q_ = out_acc->GetShape().GetDim(2).GetValues()[0];
    fmha_fwd_splitkv_combine_kernel_args.nhead_q_  = this->q_num_heads_;
    fmha_fwd_splitkv_combine_kernel_args.hdim_v_   = this->v_head_dim_;

    fmha_fwd_splitkv_combine_kernel_args.num_splits_ = this->num_splits_;

    fmha_fwd_splitkv_combine_kernel_args.stream_ = this->context_ptr_->GetStream();

    this->register_kernel_ptr_->KernelLauncher(this->GetName(), std::move(fmha_fwd_splitkv_combine_kernel_args));

    PrintToScreen(out_ptr, 3, "[" + this->op_name_ + "]" + "out_ptr");
    ResultChecker(out_ptr, std::get<0>(out_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "out_ptr");
}

template class FmhaFwdSplitKVCombineOp<ushort>;
template class FmhaFwdSplitKVCombineOp<_Float16>;

}  // namespace flashck