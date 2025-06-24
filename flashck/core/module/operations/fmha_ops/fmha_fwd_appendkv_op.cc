#include "flashck/core/module/operations/fmha_ops/fmha_fwd_appendkv_op.h"

#include "flashck/core/module/kernels/fmha_kernels/fmha_fwd_appendkv_kernel.h"

#include "flashck/core/graph/node.h"
#include "flashck/core/utils/memory_utils.h"

namespace flashck {
template<typename T>
FmhaFwdAppendKVOp<T>::FmhaFwdAppendKVOp(std::string       op_name,
                                        FmhaOperationMode op_mode,
                                        int64_t           q_num_heads,
                                        int64_t           qk_head_dim,
                                        int64_t           kv_num_heads,
                                        int64_t           v_head_dim,
                                        int64_t           rotary_dim,
                                        RopeEnum          rope_enum,
                                        int64_t           paged_block_size,
                                        bool              use_cache_batch_idx):
    FmhaCommonOp<T, FmhaFwdAppendKVOp<T>>::FmhaCommonOp(op_name)
{
    if (CppTypeToDataType<T>::Type() != DataType::FLOAT16 || CppTypeToDataType<T>::Type() != DataType::BFLOAT16) {
        LI_ENFORCE_GT(rotary_dim, 0, Unavailable("rotary embedding is only available for data type=fp16|bf16"));
    }

    LI_ENFORCE_LE(rotary_dim, qk_head_dim, Unavailable("rotary_dim should be less than or equal to head dim for q"));
    LI_ENFORCE_EQ(rotary_dim % 16, 0, Unavailable("only rotary dimensions divisible by 16 are currently supported"));
    LI_ENFORCE_EQ(paged_block_size % 128,
                  0,
                  Unavailable("only paged-kvcache block size divisible by 128 are currently supported"));
    if (paged_block_size > 0 && use_cache_batch_idx) {
        LI_THROW(Unavailable("paged-kvcache does not support cache_batch_idx"));
    }
    if (use_cache_batch_idx && op_mode == FmhaOperationMode::Group) {
        Unavailable("fmha append kv op only supports batch mode");
    }
    LI_ENFORCE_EQ(
        q_num_heads % kv_num_heads,
        0,
        Unavailable("q_num_heads should be divisible by kv_num_heads, but got {} and {}", q_num_heads, kv_num_heads));

    LI_ENFORCE_LE(qk_head_dim, 256, Unavailable("FlashAttention forward only supports head dimension at most 256"));

    this->op_kind_      = FmhaOperationKind::FwdAppendKV;
    this->op_name_      = op_name;
    this->op_mode_      = op_mode;
    this->q_num_heads_  = q_num_heads;
    this->qk_head_dim_  = qk_head_dim;
    this->kv_num_heads_ = kv_num_heads == -1 ? q_num_heads : kv_num_heads;
    this->v_head_dim_   = v_head_dim == -1 ? qk_head_dim : v_head_dim;

    this->paged_block_size_ = paged_block_size;

    this->rotary_dim_ = rotary_dim;
    this->rope_enum_  = rope_enum;

    this->use_cache_batch_idx_ = use_cache_batch_idx;
}

template<typename T>
FmhaProblem FmhaFwdAppendKVOp<T>::DefineProblemImpl(const std::vector<int64_t>& inverse_res)
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

// q (batch_size, q_seq_len, q_num_heads, qk_head_dim)
// cache_k (batch_size, kv_seq_len, kv_num_heads, qk_head_dim)
// cache_v (batch_size, kv_seq_len, kv_num_heads, v_head_dim)
// k (batch_size, new_kv_seq_len, kv_num_heads, qk_head_dim)
// v (batch_size, new_kv_seq_len, kv_num_heads, v_head_dim)

template<typename T>
std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> FmhaFwdAppendKVOp<T>::ExtractDimsImpl()
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
          std::make_shared<DimInfo>(TensorSource::kInput, 3, dim_idx_0),
          std::make_shared<DimInfo>(TensorSource::kInput, 4, dim_idx_0)}},
        {"seqlen_q", {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_1)}},
        {"seqlen_k",
         {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_1)}},
        {"seqlen_knew",
         {std::make_shared<DimInfo>(TensorSource::kInput, 3, dim_idx_1),
          std::make_shared<DimInfo>(TensorSource::kInput, 4, dim_idx_1)}},
        {"nhead_q", {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_2)}},
        {"nhead_k",
         {std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::kInput, 2, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::kInput, 3, dim_idx_2),
          std::make_shared<DimInfo>(TensorSource::kInput, 4, dim_idx_2)}},
        {"hdim_q",
         {std::make_shared<DimInfo>(TensorSource::kInput, 0, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::kInput, 1, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::kInput, 3, dim_idx_3)}},
        {"hdim_v",
         {std::make_shared<DimInfo>(TensorSource::kInput, 3, dim_idx_3),
          std::make_shared<DimInfo>(TensorSource::kInput, 4, dim_idx_3)}},
    };
}

template<typename T>
void FmhaFwdAppendKVOp<T>::SanityCheck(Variable* q,
                                       Variable* cache_k,
                                       Variable* cache_v,
                                       Variable* k,
                                       Variable* v,
                                       Variable* block_table,
                                       Variable* cache_batch_idx,
                                       Variable* rotary_cos,
                                       Variable* rotary_sin,
                                       Variable* cache_seqlen_k)
{
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

    // new kv seq_len
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
    }

    // k seq_len
    if (cache_seqlen_k != nullptr) {
        if (cache_seqlen_k->GetDtype() != DataType::INT64) {
            LI_THROW(Unimplemented("cache_seqlen_k must be int64 tensor, but got cache_seqlen_k: {}",
                                   DataTypeToString(cache_seqlen_k->GetDtype())));
        }

        LI_ENFORCE_EQ(cache_seqlen_k->GetShape().GetNumDim(),
                      1,
                      Unimplemented("cache_seqlen_k must be 1D tensor, but got cache_seqlen_k: {}",
                                    cache_seqlen_k->GetShape().GetNumDim()));

        if (cache_seqlen_k->GetShape().GetDim(0) != k->GetShape().GetDim(0)) {
            LI_THROW(Unimplemented("cache_seqlen_k must have the same batch size as k, but got cache_seqlen_k: {}",
                                   cache_seqlen_k->GetShape().GetDim(0).ToString()));
        }
    }

    // check cache_k, cache_value
    if (this->paged_block_size_ > 0) {
        // cache_k: [max_num_page_blocks, paged_block_size, kv_num_heads, qk_head_dim]
        // cache_v: [max_num_page_blocks, paged_block_size, kv_num_heads, v_head_dim]
        if (cache_k->GetShape().GetNumDim() != 4) {
            LI_THROW(
                Unimplemented("cache_k must have 4 dimensions, but got cache_k: {}", cache_k->GetShape().GetNumDim()));
        }

        if (cache_v->GetShape().GetNumDim() != 4) {
            LI_THROW(
                Unimplemented("cache_v must have 4 dimensions, but got cache_v: {}", cache_k->GetShape().GetNumDim()));
        }

        if (cache_k->GetShape().GetDim(0) != cache_v->GetShape().GetDim(0)
            || cache_k->GetShape().GetDim(1) != DDim(this->paged_block_size_)
            || cache_k->GetShape().GetDim(2) != DDim(this->kv_num_heads_)
            || cache_k->GetShape().GetDim(3) != DDim(this->qk_head_dim_)) {
            LI_THROW(Unimplemented("cache_k must have shape [num_blocks, paged_block_size, qk_num_heads, qk_head_dim], "
                                   "but got cache_k: {}",
                                   cache_k->GetShape().ToString()));
        }

        if (cache_v->GetShape().GetDim(1) != DDim(this->paged_block_size_)
            || cache_v->GetShape().GetDim(2) != DDim(this->kv_num_heads_)
            || cache_v->GetShape().GetDim(3) != DDim(this->v_head_dim_)) {
            LI_THROW(Unimplemented("cache_v must have shape [num_blocks, paged_block_size, qk_num_heads, v_head_dim], "
                                   "but got cache_v: {}",
                                   cache_v->GetShape().ToString()));
        }
    }
    else {
        // cache_k: [batch_size, kv_seq_len, kv_num_heads, qk_head_dim]
        // cache_v: [batch_size, kv_seq_len, kv_num_heads, v_head_dim]
        if (cache_k->GetShape().GetNumDim() != 4) {
            LI_THROW(
                Unimplemented("cache_k must have 4 dimensions, but got cache_k: {}", cache_k->GetShape().GetNumDim()));
        }

        if (cache_k->GetShape().GetNumDim() != 4) {
            LI_THROW(
                Unimplemented("cache_k must have 4 dimensions, but got cache_k: {}", cache_k->GetShape().GetNumDim()));
        }

        if (cache_k->GetShape().GetDim(0) != k->GetShape().GetDim(0)
            || cache_k->GetShape().GetDim(1) != cache_v->GetShape().GetDim(1)
            || cache_k->GetShape().GetDim(2) != DDim(this->kv_num_heads_)
            || cache_k->GetShape().GetDim(3) != DDim(this->qk_head_dim_)) {
            LI_THROW(Unimplemented("cache_k must have shape [batch_size, kv_seq_len, qk_num_heads, qk_head_dim], "
                                   "but got cache_k: {}",
                                   cache_k->GetShape().ToString()));
        }

        if (cache_v->GetShape().GetDim(0) != v->GetShape().GetDim(0)
            || cache_v->GetShape().GetDim(1) != cache_k->GetShape().GetDim(1)
            || cache_v->GetShape().GetDim(2) != DDim(this->kv_num_heads_)
            || cache_v->GetShape().GetDim(3) != DDim(this->v_head_dim_)) {
            LI_THROW(Unimplemented("cache_v must have shape [batch_size, kv_seq_len, qk_num_heads, v_head_dim], "
                                   "but got cache_v: {}",
                                   cache_v->GetShape().ToString()));
        }
    }

    // check rotary sin, rotary cos
    if (rotary_sin != nullptr && rotary_cos != nullptr) {
        if (this->rotary_dim_ <= 0) {
            LI_THROW(Unimplemented("rotary_dim must be set when using rotary_sin and rotary_cos"));
        }

        if (rotary_sin->GetShape().GetNumDim() != 2) {
            LI_THROW(Unimplemented("rotary_sin must have 2 dimension, but got rotary_sin: {}",
                                   rotary_sin->GetShape().GetNumDim()));
        }

        if (rotary_sin->GetShape().GetDim(0) != rotary_cos->GetShape().GetDim(0)
            || rotary_sin->GetShape().GetDim(1) != DDim(this->rotary_dim_ / 2)) {
            LI_THROW(Unimplemented(
                "rotary_sin must have shape [max(q_seqlen, kv_seq_len)*2, rotary_dim / 2], but got rotary_sin: {}",
                rotary_sin->GetShape().ToString()));
        }

        if (rotary_cos->GetShape().GetNumDim() != 2) {
            LI_THROW(Unimplemented("rotary_cos must have 2 dimension, but got rotary_cos: {}",
                                   rotary_cos->GetShape().GetNumDim()));
        }

        if (rotary_cos->GetShape().GetDim(1) != DDim(this->rotary_dim_ / 2)) {
            LI_THROW(Unimplemented(
                "rotary_cos must have shape [max(q_seqlen, kv_seq_len)*2, rotary_dim / 2], but got rotary_cos: {}",
                rotary_cos->GetShape().ToString()));
        }
    }

    // cache_batch_idx
    if (this->use_cache_batch_idx_) {
        if (cache_batch_idx == nullptr) {
            LI_THROW(Unimplemented("cache_batch_idx not nullptr"));
        }

        if (cache_batch_idx->GetShape().GetNumDim() != 1) {
            LI_THROW(Unimplemented("cache_batch_idx must have 1 dimension, but got cache_batch_idx: {}",
                                   cache_batch_idx->GetShape().GetNumDim()));
        }

        if (cache_batch_idx->GetShape().GetDim(0) != q->GetShape().GetDim(0)) {
            LI_THROW(Unimplemented("cache_batch_idx must have shape [batch_size], but got cache_batch_idx: {}",
                                   cache_batch_idx->GetShape().ToString()));
        }
    }
}

template<typename T>
std::tuple<Shape, Shape, Shape>
FmhaFwdAppendKVOp<T>::InferShape(Variable* q, Variable* cache_k, Variable* k, Variable* v)
{
    DDim batch_size_dim     = q->GetShape().GetDim(0);
    DDim q_seq_len_dim      = q->GetShape().GetDim(1);
    DDim kv_seq_len_dim     = cache_k->GetShape().GetDim(1);
    DDim new_kv_seq_len_dim = k->GetShape().GetDim(1);
    DDim q_num_heads_dim    = q->GetShape().GetDim(2);
    DDim kv_num_heads_dim   = v->GetShape().GetDim(2);
    DDim qk_head_dim_dim    = q->GetShape().GetDim(3);
    DDim v_head_dim_dim     = v->GetShape().GetDim(3);

    Shape q_out_shape = q->GetShape();
    Shape k_out_shape = Shape({batch_size_dim, kv_seq_len_dim, kv_num_heads_dim, qk_head_dim_dim});
    Shape v_out_shape = Shape({batch_size_dim, kv_seq_len_dim, kv_num_heads_dim, v_head_dim_dim});

    return std::make_tuple(q_out_shape, k_out_shape, v_out_shape);
}

template<typename T>
std::vector<Variable*> FmhaFwdAppendKVOp<T>::operator()(Variable* q,
                                                        Variable* cache_k,
                                                        Variable* cache_v,
                                                        Variable* k,
                                                        Variable* v,
                                                        Variable* block_table,
                                                        Variable* cache_batch_idx,
                                                        Variable* rotary_cos,
                                                        Variable* rotary_sin,
                                                        Variable* cache_seqlen_k)
{
    SanityCheck(q, cache_k, cache_v, k, v, block_table, cache_batch_idx, rotary_cos, rotary_sin, cache_seqlen_k);

    Shape q_out_shape, k_out_shape, v_out_shape;
    std::tie(q_out_shape, k_out_shape, v_out_shape) = InferShape(q, cache_k, k, v);
    auto q_max_output_size                          = std::get<1>(q_out_shape.GetElementSizeTuple());
    auto k_max_output_size                          = std::get<1>(k_out_shape.GetElementSizeTuple());
    auto v_max_output_size                          = std::get<1>(v_out_shape.GetElementSizeTuple());
    this->output_var_                               = {
        new Variable(this->op_name_ + std::string("_q_output"), q_max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_.emplace_back(
        new Variable(this->op_name_ + std::string("_k_output"), k_max_output_size, CppTypeToDataType<T>::Type()));
    this->output_var_.emplace_back(
        new Variable(this->op_name_ + std::string("_v_output"), v_max_output_size, CppTypeToDataType<T>::Type()));
    this->output_var_[0]->SetShape(q_out_shape);
    this->output_var_[1]->SetShape(k_out_shape);
    this->output_var_[2]->SetShape(v_out_shape);

    this->input_var_ = {
        q, cache_k, cache_v, k, v, block_table, cache_batch_idx, rotary_cos, rotary_sin, cache_seqlen_k};
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
std::function<std::vector<std::string>(const std::string&)> FmhaFwdAppendKVOp<T>::GenBuildCmd()
{
    auto fbuild_cmd = [&](const std::string& exec_key) {
        std::vector<int64_t> input_shape = this->InvertExecKey(exec_key);
        // [batch, nhead_k, nhead_q, hdim_q, hdim_v, seqlen_k, seqlen_q, seqlen_knew]
        std::vector<std::string> cmd_str = {"-b=" + std::to_string(input_shape[0]),
                                            "-h=" + std::to_string(input_shape[2]),
                                            "-h_k=" + std::to_string(input_shape[1]),
                                            "-s=" + std::to_string(input_shape[5]),
                                            "-s_k=" + std::to_string(input_shape[7]),
                                            "-s_knew=" + std::to_string(input_shape[6]),
                                            "-d=" + std::to_string(input_shape[3]),
                                            "-d_v=" + std::to_string(input_shape[4]),
                                            "-has_mask="
                                                + (this->mask_enum_ != GenericAttentionMaskEnum::NO_MASK
                                                           || this->window_size_ != std::array<int64_t, 2>({-1, -1}) ?
                                                       std::string("1") :
                                                       std::string("0")),
                                            "-rotary_dim=" + std::to_string(this->rotary_dim_),
                                            "-paged_block_size=" + std::to_string(this->paged_block_size_),
                                            "-use_cache_batch_idx=" + std::to_string(this->use_cache_batch_idx_)};

        return cmd_str;
    };

    return fbuild_cmd;
}

template<typename T>
void FmhaFwdAppendKVOp<T>::ForwardImpl()
{
    auto q       = this->GetParentNode(0);
    auto cache_k = this->GetParentNode(1);
    auto cache_v = this->GetParentNode(2);
    auto k       = this->GetParentNode(3);
    auto v       = this->GetParentNode(4);
    auto q_out   = this->GetChildNode(0);
    auto k_out   = this->GetChildNode(1);
    auto v_out   = this->GetChildNode(2);

    T* q_ptr       = (T*)q->GetValue();
    T* cache_k_ptr = (T*)cache_k->GetValue();
    T* cache_v_ptr = (T*)cache_v->GetValue();
    T* k_ptr       = (T*)k->GetValue();
    T* v_ptr       = (T*)v->GetValue();
    T* q_out_ptr   = (T*)q_out->GetValue();
    T* k_out_ptr   = (T*)k_out->GetValue();
    T* v_out_ptr   = (T*)v_out->GetValue();

    T* block_table_ptr = nullptr;
    if (this->paged_block_size_ > 0) {
        block_table_ptr = (T*)this->GetParentNode(5)->GetValue();
    }

    int64_t* cache_batch_idx_ptr = nullptr;
    if (this->use_cache_batch_idx_) {
        cache_batch_idx_ptr = (int64_t*)this->GetParentNode(5 + (block_table_ptr != nullptr))->GetValue();
    }

    T* rotary_cos_ptr = nullptr;
    T* rotary_sin_ptr = nullptr;
    if (this->rotary_dim_ > 0) {
        rotary_cos_ptr =
            (T*)this->GetParentNode(5 + (block_table_ptr != nullptr) + (cache_batch_idx_ptr != nullptr))->GetValue();
        rotary_sin_ptr =
            (T*)this->GetParentNode(6 + (block_table_ptr != nullptr) + (cache_batch_idx_ptr != nullptr))->GetValue();
    }

    Variable* cache_seqlen_k     = nullptr;
    int64_t*  cache_seqlen_k_ptr = nullptr;
    cache_seqlen_k     = this->GetParentNode(5 + (block_table_ptr != nullptr) + (cache_batch_idx_ptr != nullptr)
                                         + (rotary_cos_ptr != nullptr) + (rotary_sin_ptr != nullptr));
    cache_seqlen_k_ptr = (int64_t*)cache_seqlen_k->GetValue();

    // update cache_q cache_v shape
    if (this->paged_block_size_ <= 0) {
        cache_k->GetShape().SetDim(1, cache_k->GetShape().GetDim(1) + k->GetShape().GetDim(1));
        cache_v->GetShape().SetDim(1, cache_v->GetShape().GetDim(1) + v->GetShape().GetDim(1));
    }

    // update seq_len_k shape
    if (this->op_mode_ == FmhaOperationMode::Group) {
        cache_seqlen_k->GetShape().SetDim(0, k->GetShape().GetDim(0));
    }

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    PrintToScreen(q_ptr, 3, "[" + this->op_name_ + "]" + "q_ptr");
    PrintToScreen(cache_k_ptr, 3, "[" + this->op_name_ + "]" + "cache_k_ptr");
    PrintToScreen(cache_v_ptr, 3, "[" + this->op_name_ + "]" + "cache_v_ptr");
    PrintToScreen(k_ptr, 3, "[" + this->op_name_ + "]" + "k_ptr");
    PrintToScreen(v_ptr, 3, "[" + this->op_name_ + "]" + "v_ptr");

    PrintToScreen(block_table_ptr, 3, "[" + this->op_name_ + "]" + "block_table_ptr");
    PrintToScreen(cache_batch_idx_ptr, 3, "[" + this->op_name_ + "]" + "cache_batch_idx_ptr");
    PrintToScreen(rotary_cos_ptr, 3, "[" + this->op_name_ + "]" + "rotary_cos_ptr");
    PrintToScreen(rotary_sin_ptr, 3, "[" + this->op_name_ + "]" + "rotary_sin_ptr");
    PrintToScreen(
        cache_seqlen_k_ptr, q->GetShape().GetDim(0).GetValues()[0], "[" + this->op_name_ + "]" + "cache_seqlen_k_ptr");

    Shape q_out_shape, k_out_shape, v_out_shape;
    std::tie(q_out_shape, k_out_shape, v_out_shape) = this->InferShape(q, cache_k, k, v);
    this->output_var_[0]->SetShape(q_out_shape);
    this->output_var_[1]->SetShape(k_out_shape);
    this->output_var_[2]->SetShape(v_out_shape);

    VLOG(1) << "fmha appendkv fwd: " << this->op_name_;

    FmhaFwdAppendKVKernelArgs fmha_fwd_appendkv_kernel_args;
    fmha_fwd_appendkv_kernel_args.q_ptr_       = q_ptr;
    fmha_fwd_appendkv_kernel_args.cache_k_ptr_ = cache_k_ptr;
    fmha_fwd_appendkv_kernel_args.cache_v_ptr_ = cache_v_ptr;
    fmha_fwd_appendkv_kernel_args.k_ptr_       = k_ptr;
    fmha_fwd_appendkv_kernel_args.v_ptr_       = v_ptr;

    fmha_fwd_appendkv_kernel_args.cache_seqlen_k_ptr_ = cache_seqlen_k_ptr;

    fmha_fwd_appendkv_kernel_args.block_table_ptr_     = block_table_ptr;
    fmha_fwd_appendkv_kernel_args.rotary_cos_ptr_      = rotary_cos_ptr;
    fmha_fwd_appendkv_kernel_args.rotary_sin_ptr_      = rotary_sin_ptr;
    fmha_fwd_appendkv_kernel_args.cache_batch_idx_ptr_ = cache_batch_idx_ptr;

    fmha_fwd_appendkv_kernel_args.batch_    = q->GetShape().GetDim(0).GetValues()[0];
    fmha_fwd_appendkv_kernel_args.seqlen_q_ = q->GetShape().GetDim(1).GetValues()[0];
    fmha_fwd_appendkv_kernel_args.seqlen_k_ = cache_k->GetShape().GetDim(1).GetValues()[0];

    fmha_fwd_appendkv_kernel_args.nhead_q_     = this->q_num_heads_;
    fmha_fwd_appendkv_kernel_args.nhead_k_     = this->kv_num_heads_;
    fmha_fwd_appendkv_kernel_args.hdim_q_      = this->qk_head_dim_;
    fmha_fwd_appendkv_kernel_args.hdim_v_      = this->v_head_dim_;
    fmha_fwd_appendkv_kernel_args.seqlen_knew_ = k->GetShape().GetDim(1).GetValues()[0];

    fmha_fwd_appendkv_kernel_args.has_mask_ =
        this->mask_enum_ != GenericAttentionMaskEnum::NO_MASK || this->window_size_ != std::array<int64_t, 2>({-1, -1});
    auto ceildiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    auto seqlen_k_vec =
        std::vector<int64_t>(cache_seqlen_k_ptr, cache_seqlen_k_ptr + fmha_fwd_appendkv_kernel_args.batch_);
    int64_t max_seqlen_k = *std::max_element(seqlen_k_vec.begin(), seqlen_k_vec.end());
    fmha_fwd_appendkv_kernel_args.max_num_page_blocks_ =
        (this->paged_block_size_ > 0 ? fmha_fwd_appendkv_kernel_args.batch_
                                           * std::max((int64_t)1, ceildiv(max_seqlen_k, this->paged_block_size_)) :
                                       0);
    fmha_fwd_appendkv_kernel_args.paged_block_size_ = this->paged_block_size_;
    fmha_fwd_appendkv_kernel_args.rotary_dim_       = this->rotary_dim_;

    fmha_fwd_appendkv_kernel_args.stream_ = this->context_ptr_->GetStream();

    this->register_kernel_ptr_->KernelLauncher(this->GetName(), std::move(fmha_fwd_appendkv_kernel_args));

    // update cache_q cache_v shape
    if (this->paged_block_size_ <= 0) {
        cache_k->GetShape().SetDim(
            0, DDim(cache_k->GetShape().GetDim(1).GetValues()[0] + fmha_fwd_appendkv_kernel_args.seqlen_knew_));
        cache_k->GetShape().SetDim(1, k->GetShape().GetDim(0).GetValues());
        cache_v->GetShape().SetDim(
            0, DDim(cache_v->GetShape().GetDim(1).GetValues()[0] + fmha_fwd_appendkv_kernel_args.seqlen_knew_));
        cache_v->GetShape().SetDim(1, v->GetShape().GetDim(0).GetValues());
    }

    // update seq_len_k shape
    cache_seqlen_k->GetShape().SetDim(0, k->GetShape().GetDim(0).GetValues()[0]);

    // update seq_len_k value
    for (int64_t i = 0; i < k->GetShape().GetDim(0).GetValues()[0]; i++) {
        cache_seqlen_k_ptr[i] += fmha_fwd_appendkv_kernel_args.seqlen_knew_;
    }

    HipD2DCpyAsync(q_out_ptr, q_ptr, std::get<0>(q->GetShape().GetElementSizeTuple()), this->context_ptr_->GetStream());
    HipD2DCpyAsync(k_out_ptr,
                   cache_k_ptr,
                   std::get<0>(cache_k->GetShape().GetElementSizeTuple()),
                   this->context_ptr_->GetStream());
    HipD2DCpyAsync(v_out_ptr,
                   cache_v_ptr,
                   std::get<0>(cache_v->GetShape().GetElementSizeTuple()),
                   this->context_ptr_->GetStream());
}

template class FmhaFwdAppendKVOp<ushort>;
template class FmhaFwdAppendKVOp<_Float16>;

}  // namespace flashck