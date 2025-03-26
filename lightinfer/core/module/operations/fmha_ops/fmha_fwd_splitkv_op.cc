#include "lightinfer/core/module/operations/fmha_ops/fmha_fwd_splitkv_op.h"

#include "lightinfer/core/module/kernels/fmha_kernels/fmha_fwd_splitkv_kernel.h"

#include "lightinfer/core/graph/node.h"

namespace lightinfer {

template<typename T>
FmhaFwdSplitKVOp<T>::FmhaFwdSplitKVOp(std::string              op_name,
                                      FmhaOperationMode        op_mode,
                                      int64_t                  q_num_heads,
                                      int64_t                  qk_head_dim,
                                      int64_t                  kv_num_heads,
                                      int64_t                  v_head_dim,
                                      float                    qk_scale,
                                      BiasEnum                 bias_enum,
                                      std::array<int64_t, 2>   window_size,
                                      GenericAttentionMaskEnum mask_enum,
                                      int64_t                  paged_block_size,
                                      bool                     use_cache_batch_idx,
                                      int64_t                  num_splits):
    FmhaCommonOp<T, FmhaFwdSplitKVOp<T>>::FmhaCommonOp(op_name)
{
    LI_ENFORCE_EQ(paged_block_size % 128,
                  0,
                  Unavailable("only paged-kvcache block size divisible by 128 are currently supported"));

    LI_ENFORCE_LE(num_splits, 128, Unavailable("num_splits greater than 128 is not supported"));

    LI_ENFORCE_EQ(
        q_num_heads % kv_num_heads,
        0,
        Unavailable("q_num_heads should be divisible by kv_num_heads, but got {} and {}", q_num_heads, kv_num_heads));

    LI_ENFORCE_LE(qk_head_dim, 256, Unavailable("FlashAttention forward only supports head dimension at most 256"));

    this->op_kind_     = FmhaOperationKind::FwdSplitKV;
    this->op_name_     = op_name;
    this->op_mode_     = op_mode;
    this->q_num_heads_ = q_num_heads;
    this->qk_head_dim_ = qk_head_dim;
    this->scale_       = qk_scale;
    this->bias_enum_   = bias_enum;

    this->kv_num_heads_ = kv_num_heads == -1 ? q_num_heads : kv_num_heads;
    this->v_head_dim_   = v_head_dim == -1 ? qk_head_dim : v_head_dim;
    this->scale_        = qk_scale == 0.0f ? 1.0 / std::sqrt(static_cast<float>(qk_head_dim)) : qk_scale;

    this->window_size_ = window_size;
    this->mask_enum_   = mask_enum;

    this->paged_block_size_ = paged_block_size;

    this->use_cache_batch_idx_ = use_cache_batch_idx;

    this->num_splits_ = num_splits;
}

template<typename T>
FmhaProblem FmhaFwdSplitKVOp<T>::DefineProblemImpl(const std::vector<int64_t>& inverse_res)
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
std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> FmhaFwdSplitKVOp<T>::ExtractDimsImpl()
{
    std::vector<int64_t> dim_idx_0{0};
    std::vector<int64_t> dim_idx_1{1};
    std::vector<int64_t> dim_idx_2{2};
    std::vector<int64_t> dim_idx_3{3};
    std::vector<int64_t> dim_idx_4{4};

    return {
        // {"num_splits",
        //  {std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_0),
        //   std::make_shared<DimInfo>(TensorSource::Output, 1, dim_idx_0)}},
        {"batch",
         {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx_0),
          std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_0),
          std::make_shared<DimInfo>(TensorSource::Input, 2, dim_idx_0),
          std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::Output, 1, dim_idx_1)}},
        {"seqlen_q",
         {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::Output, 1, dim_idx_2)}},
        {"seqlen_k",
         {std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::Input, 2, dim_idx_1)}},
        {"nhead_q",
         {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_3)}},
        {"nhead_k",
         {std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::Input, 2, dim_idx_2)}},
        {"hdim_q",
         {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_3)}},
        {"hdim_v",
         {std::make_shared<DimInfo>(TensorSource::Input, 2, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::Output, 2, dim_idx_4)}},
    };
}

template<typename T>
void FmhaFwdSplitKVOp<T>::SanityCheck(Variable* q,
                                      Variable* k,
                                      Variable* v,
                                      Variable* bias,
                                      Variable* block_table,
                                      Variable* cache_batch_idx,
                                      Variable* seqlen_k,
                                      Variable* seqstart_q,
                                      Variable* seqstart_k)
{
    // op mode
    if (this->op_mode_ == FmhaOperationMode::Group && q->GetShape().GetDim(0) != DDim(1)
        && k->GetShape().GetDim(0) != DDim(1) && v->GetShape().GetDim(0) != DDim(1)) {
        LI_THROW(Unimplemented("group mode batch size must 1"));
    }

    // num of dimensions
    if (q->GetShape().GetNumDim() != k->GetShape().GetNumDim()
        || q->GetShape().GetNumDim() != v->GetShape().GetNumDim()) {
        LI_THROW(Unimplemented(
            "query, key, value must have the same number of dimensions, but got query: {}, key: {}, value: {}",
            q->GetShape().GetNumDim(),
            k->GetShape().GetNumDim(),
            v->GetShape().GetNumDim()));
    }

    // batch size
    if (q->GetShape().GetDim(0) != k->GetShape().GetDim(0) || q->GetShape().GetDim(0) != v->GetShape().GetDim(0)) {
        LI_THROW(Unimplemented("query, key, value must have the same batch size, but got query: {}, key: {}, value: {}",
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
                Unimplemented("alibi bias must have 4 dimensions, but got bias: {}", bias->GetShape().GetNumDim()));
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

    // seqstart_q seqstart_k
    if (seqstart_q != nullptr && seqstart_k != nullptr) {
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

        LI_ENFORCE_EQ(
            seqstart_q->GetShape().GetNumDim(),
            1,
            Unimplemented("seqstart_q must be 1D tensor, but got seqstart_q: {}", seqstart_q->GetShape().GetNumDim()));

        LI_ENFORCE_EQ(
            seqstart_k->GetShape().GetNumDim(),
            1,
            Unimplemented("seqstart_k must be 1D tensor, but got seqstart_k: {}", seqstart_k->GetShape().GetNumDim()));

        if (q->GetShape().GetDim(0) != DDim(1)) {
            LI_THROW(Unimplemented("seqstart_q and seqstart_k is only supported for group mode"));
        }
    }

    // k seq_len
    if (seqlen_k != nullptr) {
        if (seqlen_k->GetDtype() != DataType::INT64) {
            LI_THROW(Unimplemented("seqlen_k must be int64 tensor, but got seqlen_k: {}",
                                   DataTypeToString(seqlen_k->GetDtype())));
        }

        LI_ENFORCE_EQ(
            seqlen_k->GetShape().GetNumDim(),
            1,
            Unimplemented("seqlen_k must be 1D tensor, but got seqlen_k: {}", seqlen_k->GetShape().GetNumDim()));

        if (seqlen_k->GetShape().GetDim(0) != k->GetShape().GetDim(0)) {
            LI_THROW(Unimplemented("seqlen_k must have the same batch size as key, but got seqlen_k: {}",
                                   seqlen_k->GetShape().GetDim(0).ToString()));
        }
    }

    // block tables
    if (block_table != nullptr) {
        if (this->paged_block_size_ <= 0) {
            LI_THROW(Unimplemented("block_table is not supported for paged_block_size == 0"));
        }

        if (block_table->GetShape().GetNumDim() != 2) {
            LI_THROW(Unimplemented("block_table must have 2 dimensions, but got block_table: {}",
                                   block_table->GetShape().GetNumDim()));
        }

        if (block_table->GetShape().GetDim(0) != q->GetShape().GetDim(0)) {
            LI_THROW(Unimplemented("block_table must have the same batch size as query, but got block_table: {}",
                                   block_table->GetShape().GetDim(0).ToString()));
        }

        // if (block_table->GetShape().GetDim(1) != DDim(this->num_blocks_)) {
        //     LI_THROW(
        //         Unimplemented("block_table must have the same num_blocks as num_blocks, but got block_table: {}",
        //                       block_table->GetShape().GetDim(1).ToString()));
        // }
    }

    // cache_batch_idx
    if (cache_batch_idx != nullptr) {
        LI_ENFORCE_EQ(cache_batch_idx->GetShape().GetNumDim(), 1, Unavailable("cache_batch_idx dims should be 1"));

        if (cache_batch_idx->GetShape().GetDim(0) != q->GetShape().GetDim(0)) {
            LI_THROW(
                Unimplemented("cache_batch_idx must have the same batch size as query, but got cache_batch_idx: {}",
                              cache_batch_idx->GetShape().GetDim(0).ToString()));
        }
    }
}

template<typename T>
std::tuple<Shape, Shape> FmhaFwdSplitKVOp<T>::InferShape(Variable* q, Variable* v)
{
    DDim batch_size_dim = q->GetShape().GetDim(0);
    DDim seq_len_dim    = q->GetShape().GetDim(1);
    DDim num_heads_dim  = q->GetShape().GetDim(2);
    DDim head_dim       = v->GetShape().GetDim(3);

    Shape out_shape, lse_shape;
    out_shape = Shape({DDim(this->num_splits_), batch_size_dim, seq_len_dim, num_heads_dim, head_dim});
    lse_shape = Shape({DDim(this->num_splits_), batch_size_dim, seq_len_dim, num_heads_dim});

    return std::make_tuple(out_shape, lse_shape);
}

template<typename T>
std::vector<Variable*> FmhaFwdSplitKVOp<T>::operator()(Variable* q,
                                                       Variable* k,
                                                       Variable* v,
                                                       Variable* bias,
                                                       Variable* block_table,
                                                       Variable* cache_batch_idx,
                                                       Variable* seqlen_k,
                                                       Variable* seqstart_q,
                                                       Variable* seqstart_k)
{
    SanityCheck(q, k, v, bias, block_table, cache_batch_idx, seqlen_k, seqstart_q, seqstart_k);

    Shape output_shape, lse_shape;
    std::tie(output_shape, lse_shape) = InferShape(q, v);
    auto max_output_size              = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_                 = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);
    auto max_lse_size = std::get<1>(lse_shape.GetElementSizeTuple());
    this->output_var_.emplace_back(
        new Variable(this->op_name_ + std::string("_lse"), max_lse_size, CppTypeToDataType<T>::Type()));
    this->output_var_[1]->SetShape(lse_shape);

    this->input_var_ = {q, k, v, bias, block_table, cache_batch_idx, seqlen_k, seqstart_q, seqstart_k};
    std::vector<Node*> parents_node;
    for (auto var : this->input_var_) {
        if (var != nullptr) {
            parents_node.push_back(var);
        }
    }
    this->SetParentsNode(parents_node);
    std::vector<Node*> children_node;
    for (auto var : this->output_var_) {
        children_node.push_back(static_cast<Node*>(var));
    }
    this->SetChildrenNode(children_node);

    return this->output_var_;
}

template<typename T>
std::function<std::vector<std::string>(const std::string&)> FmhaFwdSplitKVOp<T>::GenBuildCmd()
{
    auto fbuild_cmd = [&](const std::string& exec_key) {
        std::vector<int64_t> input_shape = this->InvertExecKey(exec_key);
        // [batch, nhead_k, nhead_q, hdim_q, hdim_v, seqlen_k, seqlen_q]
        std::vector<std::string> cmd_str = {"-b=" + std::to_string(input_shape[0]),
                                            "-h=" + std::to_string(input_shape[2]),
                                            "-h_k=" + std::to_string(input_shape[1]),
                                            "-s=" + std::to_string(input_shape[6]),
                                            "-s_k=" + std::to_string(input_shape[5]),
                                            "-d=" + std::to_string(input_shape[3]),
                                            "-d_v=" + std::to_string(input_shape[4]),
                                            "-scale=" + std::to_string(this->scale_),
                                            "-paged_block_size=" + std::to_string(this->paged_block_size_),
                                            "-mask=" + std::to_string(static_cast<int>(this->mask_enum_)),
                                            "-window_left_size=" + std::to_string(this->window_size_[0]),
                                            "-window_right_size=" + std::to_string(this->window_size_[1])};

        return cmd_str;
    };

    return fbuild_cmd;
}

template<typename T>
void FmhaFwdSplitKVOp<T>::ForwardImpl()
{
    auto q = this->GetParentNode(0);
    auto k = this->GetParentNode(1);
    auto v = this->GetParentNode(2);

    T* out_ptr = (T*)this->GetChildNode(0)->GetValue();
    T* lse_ptr = nullptr;
    if (this->num_splits_ > 1) {
        lse_ptr = (T*)this->GetChildNode(1)->GetValue();
    }

    T* q_ptr = (T*)q->GetValue();
    T* k_ptr = (T*)k->GetValue();
    T* v_ptr = (T*)v->GetValue();

    T* bias_ptr = nullptr;
    if (this->bias_enum_ != BiasEnum::NO_BIAS) {
        bias_ptr = (T*)this->GetParentNode(3)->GetValue();
    }

    T* block_table_ptr = nullptr;
    if (this->paged_block_size_ > 0) {
        block_table_ptr = (T*)this->GetParentNode(3 + (bias_ptr != nullptr))->GetValue();
    }

    int64_t* cache_batch_idx_ptr = nullptr;
    if (this->use_cache_batch_idx_) {
        cache_batch_idx_ptr =
            (int64_t*)this->GetParentNode(3 + (bias_ptr != nullptr) + (block_table_ptr != nullptr))->GetValue();
    }

    int64_t* seqstart_q_ptr = nullptr;
    int64_t* seqstart_k_ptr = nullptr;
    int64_t* seqlen_k_ptr   = nullptr;

    seqlen_k_ptr =
        (int64_t*)this
            ->GetParentNode(3 + (bias_ptr != nullptr) + (block_table_ptr != nullptr) + (cache_batch_idx_ptr != nullptr))
            ->GetValue();

    if (this->op_mode_ == FmhaOperationMode::Group) {
        seqstart_q_ptr = (int64_t*)this
                             ->GetParentNode(3 + (bias_ptr != nullptr) + (block_table_ptr != nullptr)
                                             + (cache_batch_idx_ptr != nullptr) + (seqlen_k_ptr != nullptr))
                             ->GetValue();
        seqstart_k_ptr = (int64_t*)this
                             ->GetParentNode(4 + (bias_ptr != nullptr) + (block_table_ptr != nullptr)
                                             + (cache_batch_idx_ptr != nullptr) + (seqlen_k_ptr != nullptr))
                             ->GetValue();
    }

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    PrintToScreen(q_ptr, 3, "[" + this->op_name_ + "]" + "q_ptr");
    PrintToScreen(k_ptr, 3, "[" + this->op_name_ + "]" + "k_ptr");
    PrintToScreen(v_ptr, 3, "[" + this->op_name_ + "]" + "v_ptr");
    PrintToScreen(bias_ptr, 3, "[" + this->op_name_ + "]" + "bias_ptr");
    PrintToScreen(block_table_ptr, 3, "[" + this->op_name_ + "]" + "block_table_ptr");
    PrintToScreen(cache_batch_idx_ptr, 3, "[" + this->op_name_ + "]" + "cache_batch_idx_ptr");
    PrintToScreen(seqstart_q_ptr, 3, "[" + this->op_name_ + "]" + "seqstart_q_ptr");
    PrintToScreen(seqstart_k_ptr, 3, "[" + this->op_name_ + "]" + "seqstart_k_ptr");
    PrintToScreen(seqlen_k_ptr, 3, "[" + this->op_name_ + "]" + "seqlen_k_ptr");

    Shape out_shape, lse_shape;
    std::tie(out_shape, lse_shape) = this->InferShape(q, v);
    this->output_var_[0]->SetShape(out_shape);
    this->output_var_[1]->SetShape(lse_shape);

    VLOG(1) << "fmha Forward: " << this->op_name_ << " out_shape: " << out_shape.ToString();

    FmhaFwdSplitKVKernelArgs fmha_fwd_splitkv_kernel_args;
    fmha_fwd_splitkv_kernel_args.q_ptr_       = q_ptr;
    fmha_fwd_splitkv_kernel_args.k_ptr_       = k_ptr;
    fmha_fwd_splitkv_kernel_args.v_ptr_       = v_ptr;
    fmha_fwd_splitkv_kernel_args.bias_ptr_    = bias_ptr;
    fmha_fwd_splitkv_kernel_args.lse_acc_ptr_ = lse_ptr;
    fmha_fwd_splitkv_kernel_args.out_acc_ptr_ = out_ptr;

    fmha_fwd_splitkv_kernel_args.seqstart_q_ptr_ = seqstart_q_ptr;
    fmha_fwd_splitkv_kernel_args.seqstart_k_ptr_ = seqstart_k_ptr;
    fmha_fwd_splitkv_kernel_args.seqlen_k_ptr_   = seqlen_k_ptr;

    fmha_fwd_splitkv_kernel_args.batch_    = q->GetShape().GetDim(0).GetValues()[0];
    fmha_fwd_splitkv_kernel_args.seqlen_q_ = q->GetShape().GetDim(1).GetValues()[0];
    fmha_fwd_splitkv_kernel_args.seqlen_k_ = k->GetShape().GetDim(1).GetValues()[0];

    fmha_fwd_splitkv_kernel_args.nhead_q_ = this->q_num_heads_;
    fmha_fwd_splitkv_kernel_args.nhead_k_ = this->kv_num_heads_;
    fmha_fwd_splitkv_kernel_args.hdim_q_  = this->qk_head_dim_;
    fmha_fwd_splitkv_kernel_args.hdim_v_  = this->v_head_dim_;

    int64_t max_seqlen_q = std::numeric_limits<int64_t>::min();
    if (this->op_mode_ == FmhaOperationMode::Group) {
        for (int wb = 0; wb < fmha_fwd_splitkv_kernel_args.batch_; ++wb) {
            auto real_seqlen_q = seqstart_q_ptr[wb + 1] - seqstart_q_ptr[wb];

            if (max_seqlen_q < real_seqlen_q) {
                max_seqlen_q = real_seqlen_q;
            }
        }
    }
    else {
        max_seqlen_q = q->GetShape().GetDim(1).GetValues()[0];
    }

    fmha_fwd_splitkv_kernel_args.max_seqlen_q_ = max_seqlen_q;

    fmha_fwd_splitkv_kernel_args.scale_ = this->scale_;

    fmha_fwd_splitkv_kernel_args.window_size_ = this->window_size_;
    fmha_fwd_splitkv_kernel_args.mask_type_   = static_cast<uint32_t>(this->mask_enum_);

    auto    ceildiv      = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    auto    seqlen_k_vec = std::vector<int64_t>(seqlen_k_ptr, seqlen_k_ptr + fmha_fwd_splitkv_kernel_args.batch_);
    int64_t max_seqlen_k = *std::max_element(seqlen_k_vec.begin(), seqlen_k_vec.end());
    fmha_fwd_splitkv_kernel_args.max_num_page_blocks_ =
        (this->paged_block_size_ > 0 ? fmha_fwd_splitkv_kernel_args.batch_
                                           * std::max((int64_t)1, ceildiv(max_seqlen_k, this->paged_block_size_)) :
                                       0);

    fmha_fwd_splitkv_kernel_args.paged_block_size_ = this->paged_block_size_;

    fmha_fwd_splitkv_kernel_args.stream_ = this->context_ptr_->GetStream();

    this->register_kernel_ptr_->KernelLauncher(this->GetName(), std::move(fmha_fwd_splitkv_kernel_args));

    PrintToScreen(out_ptr, 3, "[" + this->op_name_ + "]" + "out_ptr");
    ResultChecker(out_ptr, std::get<0>(out_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "out_ptr");
    PrintToScreen(lse_ptr, 3, "[" + this->op_name_ + "]" + "lse_ptr");
    ResultChecker(lse_ptr, std::get<0>(lse_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "lse_ptr");
}

template class FmhaFwdSplitKVOp<ushort>;
template class FmhaFwdSplitKVOp<_Float16>;

}  // namespace lightinfer