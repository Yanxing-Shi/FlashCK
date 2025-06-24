#include "flashck/core/module/operations/fmha_ops/fmha_fwd_op.h"

#include "flashck/core/module/kernels/fmha_kernels/fmha_fwd_kernel.h"

#include "flashck/core/graph/node.h"

namespace flashck {

template<typename T>
FmhaFwdOp<T>::FmhaFwdOp(std::string              op_name,
                        FmhaOperationMode        op_mode,
                        int64_t                  q_num_heads,
                        int64_t                  qk_head_dim,
                        int64_t                  kv_num_heads,
                        int64_t                  v_head_dim,
                        float                    qk_scale,
                        BiasEnum                 bias_enum,
                        std::array<int64_t, 2>   window_size,
                        GenericAttentionMaskEnum mask_enum,
                        bool                     is_packed_qkv):
    FmhaCommonOp<T, FmhaFwdOp<T>>::FmhaCommonOp(op_name)
{
    this->op_kind_     = FmhaOperationKind::Fwd;
    this->op_name_     = op_name;
    this->op_mode_     = op_mode;
    this->q_num_heads_ = q_num_heads;
    this->qk_head_dim_ = qk_head_dim;
    this->scale_       = qk_scale;
    this->bias_enum_   = bias_enum;
    this->mask_enum_   = mask_enum;

    this->kv_num_heads_ = kv_num_heads == -1 ? q_num_heads : kv_num_heads;
    this->v_head_dim_   = v_head_dim == -1 ? qk_head_dim : v_head_dim;
    this->scale_        = qk_scale == 0.0f ? 1.0 / std::sqrt(static_cast<float>(qk_head_dim)) : qk_scale;
    this->window_size_  = window_size;

    this->is_packed_qkv_ = is_packed_qkv;

    LI_ENFORCE_EQ(
        q_num_heads % kv_num_heads,
        0,
        Unavailable("q_num_heads should be divisible by kv_num_heads, but got {} and {}", q_num_heads, kv_num_heads));

    LI_ENFORCE_LE(qk_head_dim, 256, Unavailable("FlashAttention forward only supports head dimension at most 256"));
}

template<typename T>
FmhaProblem FmhaFwdOp<T>::DefineProblemImpl(const std::vector<int64_t>& inverse_res)
{
    FmhaProblem fmha_problem{CppTypeToDataType<T>::Type(),
                             this->op_mode_,
                             this->op_kind_,
                             this->mask_enum_,
                             this->window_size_,
                             this->bias_enum_,
                             false,  // static quant
                             inverse_res[0],
                             inverse_res[6],
                             inverse_res[6],
                             inverse_res[5],
                             inverse_res[1],
                             inverse_res[2],
                             inverse_res[3],
                             inverse_res[4],
                             this->paged_block_size_,
                             this->rope_enum_,
                             this->rotary_dim_,
                             this->num_splits_};
    return fmha_problem;
}

template<typename T>
// (B, M, Hq, K) * (B, N, Hkv, K) = (B, M, Hq, Hkv)
// softmax on (B, M, Hq, Hkv)
// (B, M, Hq, Hkv) * (B, N, Hkv, Kv) = (B, M, Hq, Kv)
std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> FmhaFwdOp<T>::ExtractDimsImpl()
{
    std::vector<int64_t> dim_idx_0{0};
    std::vector<int64_t> dim_idx_1{1};
    std::vector<int64_t> dim_idx_2{2};
    std::vector<int64_t> dim_idx_3{3};

    return {
        {"batch",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_0),
          std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_0),
          std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_0),
          std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_0)}},
        {"seqlen_q",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_1)}},
        {"seqlen_k",
         {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_1)}},
        {"nhead_q",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::kOutput, 0, dim_idx_2)}},
        {"nhead_k",
         {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_2)}},
        {"hdim_q",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_3)}},
        {"hdim_v",
         {std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::kOutput, 2, dim_idx_3)}},
    };
}

template<typename T>
void FmhaFwdOp<T>::SanityCheck(Variable* q,
                               Variable* k,
                               Variable* v,
                               Variable* bias,
                               Variable* seqstart_q,
                               Variable* seqstart_k,
                               Variable* seqlen_k)
{
    // op mode
    if (this->op_mode_ == FmhaOperationMode::Group && q->GetShape().GetDim(0) != DDim(1)) {
        LI_THROW(Unimplemented("group mode batch size must 1"));
    }

    // num of dimensions
    if (q->GetShape().GetNumDim() != k->GetShape().GetNumDim()
        || q->GetShape().GetNumDim() != v->GetShape().GetNumDim()) {
        LI_THROW(Unimplemented("q, k, v must have the same number of dimensions, but got q: {}, k: {}, v: {}",
                               q->GetShape().GetNumDim(),
                               k->GetShape().GetNumDim(),
                               v->GetShape().GetNumDim()));
    }

    // batch size
    if (q->GetShape().GetDim(0) != k->GetShape().GetDim(0) || q->GetShape().GetDim(0) != v->GetShape().GetDim(0)) {
        LI_THROW(Unimplemented("q, k, v must have the same batch size, but got q: {}, k: {}, v: {}",
                               q->GetShape().GetDim(0).ToString(),
                               k->GetShape().GetDim(0).ToString(),
                               v->GetShape().GetDim(0).ToString()));
    }

    // seq_len
    if (k->GetShape().GetDim(1) != v->GetShape().GetDim(1)) {
        LI_THROW(Unimplemented("k, v must have the same seq_len, but got k: {}, v: {}",
                               k->GetShape().GetDim(1).ToString(),
                               v->GetShape().GetDim(1).ToString()));
    }

    // num heads
    if (q->GetShape().GetDim(2) != DDim(this->q_num_heads_) || k->GetShape().GetDim(2) != DDim(this->kv_num_heads_)
        || v->GetShape().GetDim(2) != DDim(this->kv_num_heads_)) {
        LI_THROW(Unimplemented("num heads not right"));
    }

    // embed_dim
    if (q->GetShape().GetDim(3) != DDim(this->qk_head_dim_) || k->GetShape().GetDim(3) != DDim(this->qk_head_dim_)
        || v->GetShape().GetDim(3) != DDim(this->v_head_dim_)) {
        LI_THROW(Unimplemented("embedding dim not right"));
    }

    // bias
    if (bias != nullptr && this->bias_enum_ != BiasEnum::NO_BIAS) {
        if (bias->GetShape().GetNumDim() != 4 && this->bias_enum_ == BiasEnum::ELEMENTWISE_BIAS) {
            LI_THROW(Unimplemented("elementwise bias must have 4 dimensions, but got bias: {}",
                                   bias->GetShape().GetNumDim()));
        }

        if (bias->GetShape().GetNumDim() != 2 && this->bias_enum_ == BiasEnum::ALIBI) {
            LI_THROW(
                Unimplemented("alibi bias must have 2 dimensions, but got bias: {}", bias->GetShape().GetNumDim()));
        }

        Shape bias_shape      = bias->GetShape();
        this->bias_rank_info_ = this->GetBiasRankInfo(this->bias_enum_, q, k, bias);

        Shape bias_expected_shape = this->bias_enum_ == BiasEnum::ELEMENTWISE_BIAS ?
                                        Shape{q->GetShape().GetDim(0),
                                              DDim(this->q_num_heads_),
                                              q->GetShape().GetDim(1),
                                              k->GetShape().GetDim(1)} :
                                        Shape{q->GetShape().GetDim(0), DDim(this->q_num_heads_)};
        bool  broadcastable;
        Shape bias_broadcast_shape;
        std::tie(broadcastable, bias_broadcast_shape) = Shape::GetBroadCastMaxShape(bias_shape, bias_expected_shape);
        LI_ENFORCE_EQ(broadcastable,
                      true,
                      Unimplemented("bias shape is not broadcastable: {} vs {}",
                                    bias_shape.ToString(),
                                    bias_expected_shape.ToString()));
    }

    // seqstart_q
    if (seqstart_q != nullptr && seqstart_k != nullptr) {
        if (this->op_mode_ != FmhaOperationMode::Group) {
            LI_THROW(Unimplemented("seqstart_q and seqstart_k are only used in group mode"));
        }

        if (seqstart_q->GetShape().GetDim(0) != (q->GetShape().GetDim(0) + DDim(1))) {
            LI_THROW(Unimplemented("seqstart_q must have the shape [B+1], but got seqstart_q: {}",
                                   seqstart_q->GetShape().ToString()));
        }

        if (seqstart_k->GetShape().GetDim(0) != (k->GetShape().GetDim(0) + DDim(1))) {
            LI_THROW(Unimplemented("seqstart_k must have the shape [B+1], but got seqstart_k: {}",
                                   seqstart_k->GetShape().ToString()));
        }

        // if (seqstart_q->GetDtype() != DataType::INT32) {
        //     LI_THROW(Unimplemented("seqstart_q must be int32 tensor, but got seqstart_q: {}",
        //                              DataTypeToString(seqstart_q->GetDtype())));
        // }

        // if (seqstart_k->GetDtype() != DataType::INT32) {
        //     LI_THROW(Unimplemented("seqstart_k must be int32 tensor, but got seqstart_k: {}",
        //                              DataTypeToString(seqstart_k->GetDtype())));
        // }

        if (seqstart_q->GetShape().GetNumDim() != 1) {
            LI_THROW(Unimplemented("seqstart_q must be 1D, but got {}", seqstart_q->GetShape().GetNumDim()));
        }

        if (seqstart_k->GetShape().GetNumDim() != 1) {
            LI_THROW(Unimplemented("seqstart_k must be 1D, but got {}", seqstart_k->GetShape().GetNumDim()));
        }
    }

    // k seq_len
    if (seqlen_k != nullptr) {
        // if (seqlen_k->GetDtype() != DataType::INT32) {
        //     LI_THROW(Unimplemented("seqlen_k must be int32 tensor, but got seqlen_k: {}",
        //                              DataTypeToString(seqlen_k->GetDtype())));
        // }

        LI_ENFORCE_EQ(
            seqlen_k->GetShape().GetNumDim(),
            1,
            Unimplemented("seqlen_k must be 1D tensor, but got seqlen_k: {}", seqlen_k->GetShape().GetNumDim()));

        if (seqlen_k->GetShape().GetDim(0) != k->GetShape().GetDim(0)) {
            LI_THROW(Unimplemented("seqlen_k must have the same batch size as k, but got seqlen_k: {}",
                                   seqlen_k->GetShape().GetDim(0).ToString()));
        }
    }
}

template<typename T>
Shape FmhaFwdOp<T>::InferShape(Variable* q, Variable* v)
{
    DDim batch_size      = q->GetShape().GetDim(0);
    DDim q_seq_len_dim   = q->GetShape().GetDim(1);
    DDim q_num_heads_dim = q->GetShape().GetDim(2);
    DDim v_head_dim      = v->GetShape().GetDim(3);

    return Shape({batch_size, q_seq_len_dim, q_num_heads_dim, v_head_dim});
}

template<typename T>
Variable* FmhaFwdOp<T>::operator()(Variable* q,
                                   Variable* k,
                                   Variable* v,
                                   Variable* bias,
                                   Variable* seqstart_q,
                                   Variable* seqstart_k,
                                   Variable* seqlen_k)
{
    SanityCheck(q, k, v, bias, seqstart_q, seqstart_k, seqlen_k);

    Shape output_shape    = InferShape(q, v);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_     = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->input_var_ = {q, k, v, bias, seqstart_q, seqstart_k, seqlen_k};
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
std::function<std::vector<std::string>(const std::string&)> FmhaFwdOp<T>::GenBuildCmd()
{
    auto fbuild_cmd = [&](const std::string& exec_key) {
        std::vector<int64_t>     input_shape = this->InvertExecKey(exec_key);
        std::vector<std::string> cmd_str     = {"-b=" + std::to_string(input_shape[0]),
                                                "-h=" + std::to_string(input_shape[2]),
                                                "-h_k=" + std::to_string(input_shape[1]),
                                                "-s=" + std::to_string(input_shape[5]),
                                                "-s_k=" + std::to_string(input_shape[6]),
                                                "-d=" + std::to_string(input_shape[3]),
                                                "-d_v=" + std::to_string(input_shape[4]),
                                                "-scale=" + std::to_string(this->scale_),
                                                "-mask=" + std::to_string(static_cast<int>(this->mask_enum_)),
                                                "-window_left_size=" + std::to_string(this->window_size_[0]),
                                                "-window_right_size=" + std::to_string(this->window_size_[1])};

        return cmd_str;
    };

    return fbuild_cmd;
}

template<typename T>
void FmhaFwdOp<T>::ForwardImpl()
{
    auto q   = this->GetParentNode(0);
    auto k   = this->GetParentNode(1);
    auto v   = this->GetParentNode(2);
    auto out = this->GetChildNode(0);

    T* q_ptr   = (T*)q->GetValue();
    T* k_ptr   = (T*)k->GetValue();
    T* v_ptr   = (T*)v->GetValue();
    T* out_ptr = (T*)out->GetValue();

    T* bias_ptr = nullptr;
    if (this->bias_enum_ != BiasEnum::NO_BIAS) {
        bias_ptr = (T*)this->GetParentNode(3)->GetValue();
    }

    int64_t* seqstart_q_ptr = nullptr;
    int64_t* seqstart_k_ptr = nullptr;
    int64_t* seqlen_k_ptr   = nullptr;
    if (this->op_mode_ == FmhaOperationMode::Group) {
        seqstart_q_ptr = (int64_t*)this->GetParentNode(3 + (bias_ptr != nullptr))->GetValue();
        seqstart_k_ptr = (int64_t*)this->GetParentNode(4 + (bias_ptr != nullptr))->GetValue();
        seqlen_k_ptr   = (int64_t*)this->GetParentNode(5 + (bias_ptr != nullptr))->GetValue();
    }

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    // PrintToScreen(q_ptr, 3, "[" + this->op_name_ + "]" + "q_ptr");
    // PrintToScreen(k_ptr, 3, "[" + this->op_name_ + "]" + "k_ptr");
    // PrintToScreen(v_ptr, 3, "[" + this->op_name_ + "]" + "v_ptr");

    // PrintToScreen(bias_ptr, 3, "[" + this->op_name_ + "]" + "bias_ptr");
    // PrintToScreen(seqstart_q_ptr, 3, "[" + this->op_name_ + "]" + "seqstart_q_ptr");
    // PrintToScreen(seqstart_k_ptr, 3, "[" + this->op_name_ + "]" + "seqstart_k_ptr");
    // PrintToScreen(seqlen_k_ptr, 3, "[" + this->op_name_ + "]" + "seqlen_k_ptr");

    // extract actual shape
    if (this->is_packed_qkv_) {
        // packed_qkv
        if (q->GetShape().GetDim(1).GetValues()[0] != k->GetShape().GetDim(1).GetValues()[0]
            || this->qk_head_dim_ != this->v_head_dim_) {
            LI_THROW(Unavailable("qkv packed unavaliable"));
        }
        auto permute_qkv_shape = this->GetParentNode(0)->GetAncestor()->GetShape();  // [B, seqlen, 3*hidden_dim]
        VLOG(1) << "permute_qkv_shape: " << permute_qkv_shape.ToString();
        auto batch_size = permute_qkv_shape.GetDim(0).GetValues()[0];
        auto seq_len    = permute_qkv_shape.GetDim(1).GetValues()[0];
        auto hidden_dim = permute_qkv_shape.GetDim(2).GetValues()[0] / 3;

        auto offset = batch_size * seq_len * hidden_dim;
        auto shape  = Shape({batch_size, seq_len, this->q_num_heads_, hidden_dim / this->q_num_heads_});
        q->SetOffset(0, shape);
        k->SetOffset(offset, shape);
        v->SetOffset(2 * offset, shape);
        out->SetShape(Shape({batch_size, seq_len, hidden_dim}));
    }
    else {
        Shape out_shape = InferShape(q, v);
        out->SetShape(out_shape);
    }

    VLOG(1) << "fmha fwd: " << this->op_name_ << " out_shape: " << out->GetShape().ToString();

    FmhaFwdKernelArgs fmha_fwd_kernel_args;
    fmha_fwd_kernel_args.q_ptr_    = q_ptr;
    fmha_fwd_kernel_args.k_ptr_    = k_ptr;
    fmha_fwd_kernel_args.v_ptr_    = v_ptr;
    fmha_fwd_kernel_args.bias_ptr_ = bias_ptr;
    fmha_fwd_kernel_args.out_ptr_  = out_ptr;

    fmha_fwd_kernel_args.seqstart_q_ptr_ = seqstart_q_ptr;
    fmha_fwd_kernel_args.seqstart_k_ptr_ = seqstart_k_ptr;
    fmha_fwd_kernel_args.seqlen_k_ptr_   = seqlen_k_ptr;

    fmha_fwd_kernel_args.batch_    = q->GetShape().GetDim(0).GetValues()[0];
    fmha_fwd_kernel_args.seqlen_q_ = q->GetShape().GetDim(1).GetValues()[0];
    fmha_fwd_kernel_args.seqlen_k_ = k->GetShape().GetDim(1).GetValues()[0];

    fmha_fwd_kernel_args.nhead_q_ = this->q_num_heads_;
    fmha_fwd_kernel_args.nhead_k_ = this->kv_num_heads_;
    fmha_fwd_kernel_args.hdim_q_  = this->qk_head_dim_;
    fmha_fwd_kernel_args.hdim_v_  = this->v_head_dim_;

    int max_seqlen_q = std::numeric_limits<int32_t>::min();
    if (this->op_mode_ == FmhaOperationMode::Group) {
        for (int wb = 0; wb < fmha_fwd_kernel_args.batch_; ++wb) {
            auto real_seqlen_q = seqstart_q_ptr[wb + 1] - seqstart_q_ptr[wb];

            if (max_seqlen_q < real_seqlen_q) {
                max_seqlen_q = real_seqlen_q;
            }
        }
    }
    else {
        max_seqlen_q = q->GetShape().GetDim(1).GetValues()[0];
    }

    fmha_fwd_kernel_args.max_seqlen_q_ = max_seqlen_q;

    fmha_fwd_kernel_args.scale_ = this->scale_;

    fmha_fwd_kernel_args.window_size_ = this->window_size_;
    fmha_fwd_kernel_args.mask_type_   = static_cast<uint32_t>(this->mask_enum_);

    fmha_fwd_kernel_args.stream_ = this->context_ptr_->GetStream();

    std::stringstream ss;
    ss << fmha_fwd_kernel_args;
    VLOG(1) << ss.str();
    this->register_kernel_ptr_->KernelLauncher(this->GetName(), std::move(fmha_fwd_kernel_args));

    // PrintToScreen(out_ptr, 3, "[" + this->op_name_ + "]" + "out_ptr");
    // ResultChecker(out_ptr, std::get<0>(out_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "out_ptr");
}

template class FmhaFwdOp<ushort>;
template class FmhaFwdOp<_Float16>;
}  // namespace flashck