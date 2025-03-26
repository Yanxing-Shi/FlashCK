#include "lightinfer/core/module/operations/embedding_ops/embedding_add_add_layer_norm_op.h"

namespace lightinfer {

template<typename T>
EmbeddingAddAddLayerNormOp<T>::EmbeddingAddAddLayerNormOp(int64_t     vocab_size,
                                                          int64_t     type_vocab_size,
                                                          int64_t     max_position_embeddings,
                                                          int64_t     embedding_dims,
                                                          float       epsilon,
                                                          std::string op_name):
    EmbeddingCommonOp<T, EmbeddingAddAddLayerNormOp<T>>::EmbeddingCommonOp(op_name)

{
    this->vocab_size_              = vocab_size;
    this->type_vocab_size_         = type_vocab_size;
    this->max_position_embeddings_ = max_position_embeddings;
    this->embedding_dims_          = embedding_dims;
    this->epsilon_                 = epsilon;
    this->op_name_                 = op_name;
    this->epilogue_op_             = TensorOperation::AddAddLayerNorm;
}

template<typename T>
EmbeddingProblem EmbeddingAddAddLayerNormOp<T>::GetEmbeddingProblem(const int64_t num_indices)
{
    auto embedding_problem = EmbeddingProblem(num_indices,
                                              this->vocab_size_,
                                              this->type_vocab_size_,
                                              this->max_position_embeddings_,
                                              this->embedding_dims_,
                                              CppTypeToDataType<T>::Type(),
                                              DataType::INT64,
                                              CppTypeToDataType<T>::Type(),
                                              CppTypeToDataType<T>::Type(),
                                              DataType::FLOAT32,
                                              CppTypeToDataType<T>::Type(),
                                              this->epilogue_op_);
    return embedding_problem;
}

template<typename T>
EmbeddingQueryEntry EmbeddingAddAddLayerNormOp<T>::GetEmbeddingQueryEntry(const std::string& workload)
{
    auto query = EmbeddingQueryEntry(this->vocab_size_,
                                     this->type_vocab_size_,
                                     this->max_position_embeddings_,
                                     this->embedding_dims_,
                                     DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                     DataTypeToShortString(DataType::INT64),
                                     DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                     DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                     DataTypeToShortString(DataType::FLOAT32),
                                     DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                     this->op_name_,
                                     g_short_tensor_operation_names_map.at(this->epilogue_op_),
                                     Target::Instance()->GetTargetDeviceName(),
                                     SHA1ToHexString(workload));
    return query;
}

template<typename T>
std::vector<std::string> EmbeddingAddAddLayerNormOp<T>::GenOpProfileCmd(const std::string& profiler_prefix,
                                                                        const std::string& profiler_filename,
                                                                        const int64_t      workload)
{
    std::filesystem::path exe_path = std::filesystem::path(profiler_prefix) / profiler_filename;

    if (!CheckWithRetries(exe_path, 3, 5)) {
        LI_THROW(Fatal("Profiler {} is not executable", exe_path.string()));
    }

    return {exe_path.string(),
            std::to_string(workload),
            std::to_string(this->vocab_size_),
            std::to_string(this->type_vocab_size_),
            std::to_string(this->max_position_embeddings_),
            std::to_string(this->embedding_dims_)};
}

template<typename T>
Variable* EmbeddingAddAddLayerNormOp<T>::operator()(Variable* input_ids,
                                                    Variable* token_type_ids,
                                                    Variable* position_ids,
                                                    Variable* word_embeddings,
                                                    Variable* token_type_embeddings,
                                                    Variable* position_embeddings,
                                                    Variable* gamma,
                                                    Variable* beta)
{
    this->input_var_ = {input_ids,
                        token_type_ids,
                        position_ids,
                        word_embeddings,
                        token_type_embeddings,
                        position_embeddings,
                        gamma,
                        beta};

    Shape output_shape    = this->InferShape(input_ids);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_     = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->SetParentsNode({input_ids,
                          token_type_ids,
                          position_ids,
                          word_embeddings,
                          token_type_embeddings,
                          position_embeddings,
                          gamma,
                          beta});
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template<typename T>
void EmbeddingAddAddLayerNormOp<T>::ForwardImpl()
{
    int64_t* index_x0_ptr = (int64_t*)this->GetParentNode(0)->GetValue();
    int64_t* index_x1_ptr = (int64_t*)this->GetParentNode(1)->GetValue();
    int64_t* index_x2_ptr = (int64_t*)this->GetParentNode(2)->GetValue();
    T*       emb_x0_ptr   = (T*)this->GetParentNode(3)->GetValue();
    T*       emb_x1_ptr   = (T*)this->GetParentNode(4)->GetValue();
    T*       emb_x2_ptr   = (T*)this->GetParentNode(5)->GetValue();
    T*       gamma_ptr    = (T*)this->GetParentNode(6)->GetValue();
    T*       beta_ptr     = (T*)this->GetParentNode(7)->GetValue();

    T* y_ptr = (T*)this->GetChildNode(0)->GetValue();

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    // PrintToScreen(index_x0_ptr, 3, "[" + this->op_name_ + "]" + "index_x0_ptr");
    // PrintToScreen(index_x1_ptr, 3, "[" + this->op_name_ + "]" + "index_x1_ptr");
    // PrintToScreen(index_x2_ptr, 3, "[" + this->op_name_ + "]" + "index_x2_ptr");
    // PrintToScreen(emb_x0_ptr, 3, "[" + this->op_name_ + "]" + "emb_x0_ptr");
    // PrintToScreen(emb_x1_ptr, 3, "[" + this->op_name_ + "]" + "emb_x1_ptr");
    // PrintToScreen(emb_x2_ptr, 3, "[" + this->op_name_ + "]" + "emb_x2_ptr");
    // PrintToScreen(gamma_ptr, 3, "[" + this->op_name_ + "]" + "gamma_ptr");
    // PrintToScreen(beta_ptr, 3, "[" + this->op_name_ + "]" + "beta_ptr");

    Shape y_shape = this->InferShape(this->GetParentNode(0));
    this->output_var_[0]->SetShape(y_shape);

    auto indices_values = std::get<0>(this->ComputeIndicesValues(this->GetParentNode(0)));

    VLOG(1) << "EmbeddingAddLayerNormOp Forward: " << this->op_name_ << " y_shape: " << y_shape.ToString()
            << " indices_values: " << indices_values;

    EmbeddingKernelArgs kernel_args;
    kernel_args.emb_x0_ptr_ = emb_x0_ptr;
    kernel_args.emb_x1_ptr_ = emb_x1_ptr;
    kernel_args.emb_x2_ptr_ = emb_x2_ptr;

    kernel_args.index_x0_ptr_ = index_x0_ptr;
    kernel_args.index_x1_ptr_ = index_x1_ptr;
    kernel_args.index_x2_ptr_ = index_x2_ptr;

    kernel_args.gamma_ptr_ = gamma_ptr;
    kernel_args.beta_ptr_  = beta_ptr;
    kernel_args.y_ptr_     = y_ptr;

    kernel_args.embedding_dims_ = this->embedding_dims_;
    kernel_args.num_indices_    = indices_values;

    kernel_args.epsilon_ = this->epsilon_;
    kernel_args.stream_  = this->context_ptr_->GetStream();

    this->register_kernel_ptr_->KernelLauncher(this->GetName(), std::move(kernel_args));
    // PrintToScreen(y_ptr, 3, "[" + this->op_name_ + "]" + "y_ptr");
    // ResultChecker(out_ptr, std::get<0>(out_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "out_ptr");
}

template class EmbeddingAddAddLayerNormOp<float>;
template class EmbeddingAddAddLayerNormOp<_Float16>;
template class EmbeddingAddAddLayerNormOp<ushort>;
}  // namespace lightinfer