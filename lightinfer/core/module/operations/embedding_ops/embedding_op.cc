#include "lightinfer/core/module/operations/embedding_ops/embedding_op.h"

namespace lightinfer {

template<typename T>
EmbeddingOp<T>::EmbeddingOp(int64_t num_embeddings_, int64_t embedding_dims, float epsilon, std::string op_name):
    EmbeddingCommonOp<T, EmbeddingOp<T>>::EmbeddingCommonOp(op_name)
{
    this->num_embeddings_ = num_embeddings_;
    this->embedding_dims_ = embedding_dims;
    this->epsilon_        = epsilon;
    this->op_name_        = op_name;
    this->epilogue_op_    = TensorOperation::PassThrough;
}

template<typename T>
EmbeddingProblem EmbeddingOp<T>::GetEmbeddingProblem(const int64_t num_indices)
{
    auto embedding_problem = EmbeddingProblem(num_indices,
                                              this->num_embeddings_,
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
EmbeddingQueryEntry EmbeddingOp<T>::GetEmbeddingQueryEntry(const std::string& workload)
{
    auto query = EmbeddingQueryEntry(this->num_embeddings_,
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
std::vector<std::string> EmbeddingOp<T>::GenOpProfileCmd(const std::string& profiler_prefix,
                                                         const std::string& profiler_filename,
                                                         const int64_t      workload)
{
    std::filesystem::path exe_path = std::filesystem::path(profiler_prefix) / profiler_filename;

    if (!CheckWithRetries(exe_path, 3, 5)) {
        LI_THROW(Fatal("Profiler {} is not executable", exe_path.string()));
    }

    return {exe_path.string(),
            std::to_string(workload),
            std::to_string(this->num_embeddings_),
            std::to_string(this->embedding_dims_)};
}

template<typename T>
Variable* EmbeddingOp<T>::operator()(Variable* x, Variable* weight, Variable* gamma, Variable* beta)
{
    this->input_var_ = {x, weight, gamma, beta};

    Shape output_shape    = this->InferShape(x);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    this->output_var_     = {
        new Variable(this->op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    this->output_var_[0]->SetShape(output_shape);

    this->SetParentsNode({x, weight, gamma, beta});
    this->SetChildrenNode({static_cast<Node*>(this->output_var_[0])});

    return this->output_var_[0];
}

template<typename T>
void EmbeddingOp<T>::ForwardImpl()
{
    int64_t* index_x0_ptr = (int64_t*)this->GetParentNode(0)->GetValue();
    T*       emb_x0_ptr   = (T*)this->GetParentNode(1)->GetValue();
    T*       gamma_ptr    = (T*)this->GetParentNode(2)->GetValue();
    T*       beta_ptr     = (T*)this->GetParentNode(3)->GetValue();

    T* y_ptr = (T*)this->GetChildNode(0)->GetValue();

    if (!this->context_ptr_->IsBuilt()) {
        return;
    }

    // PrintToScreen(index_x0_ptr, 3, "[" + this->op_name_ + "]" + "index_x0_ptr");
    // PrintToScreen(emb_x0_ptr, 3, "[" + this->op_name_ + "]" + "emb_x0_ptr");
    // PrintToScreen(gamma_ptr, 3, "[" + this->op_name_ + "]" + "gamma_ptr");
    // PrintToScreen(beta_ptr, 3, "[" + this->op_name_ + "]" + "beta_ptr");

    Shape y_shape = this->InferShape(this->GetParentNode(0));
    this->output_var_[0]->SetShape(y_shape);

    auto indices_values = std::get<0>(this->ComputeIndicesValues(this->GetParentNode(0)));

    VLOG(1) << "EmbeddingAddLayerNormOp Forward: " << this->op_name_ << " y_shape: " << y_shape.ToString()
            << " indices_values: " << indices_values;

    EmbeddingKernelArgs kernel_args;
    kernel_args.emb_x0_ptr_ = emb_x0_ptr;

    kernel_args.index_x0_ptr_ = index_x0_ptr;

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

template class EmbeddingOp<float>;
template class EmbeddingOp<_Float16>;
template class EmbeddingOp<ushort>;
}  // namespace lightinfer