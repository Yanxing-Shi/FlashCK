#include "lightinfer/core/module/layers/embedding_layers/embedding_layer.h"

namespace lightinfer {

template<typename T>
EmbeddingLayer<T>::EmbeddingLayer(int64_t vocab_size,
                                  int64_t type_vocab_size,
                                  int64_t max_position_embeddings,
                                  int64_t embedding_dims,
                                  float   epsilon):
    Layer("EmbeddingLayer"),
    vocab_size_(vocab_size),
    type_vocab_size_(type_vocab_size),
    max_position_embeddings_(max_position_embeddings),
    embedding_dims_(embedding_dims),
    embedding_add_add_layer_norm_op_(std::make_unique<EmbeddingAddAddLayerNormOp<T>>(
        vocab_size, type_vocab_size, max_position_embeddings, embedding_dims, epsilon))
{

    // param node
    word_embeddings_var_       = std::make_unique<Variable>("word_embeddings_var", CppTypeToDataType<T>::Type());
    token_type_embeddings_var_ = std::make_unique<Variable>("token_type_embeddings_var", CppTypeToDataType<T>::Type());
    position_embeddings_var_   = std::make_unique<Variable>("position_embeddings_var", CppTypeToDataType<T>::Type());
    gamma_var_                 = std::make_unique<Variable>("gamma_var", CppTypeToDataType<T>::Type());
    beta_var_                  = std::make_unique<Variable>("beta_var", CppTypeToDataType<T>::Type());

    word_embeddings_var_->SetShape(Shape({vocab_size_, embedding_dims_}));
    token_type_embeddings_var_->SetShape(Shape({type_vocab_size_, embedding_dims_}));
    position_embeddings_var_->SetShape(Shape({max_position_embeddings_, embedding_dims_}));
    gamma_var_->SetShape(Shape({embedding_dims_}));
    beta_var_->SetShape(Shape({embedding_dims_}));

    this->context_ptr_->ExitLayer();
}

template<typename T>
EmbeddingLayer<T>::EmbeddingLayer(int64_t num_embeddings, int64_t embedding_dims, float epsilon):
    Layer("EmbeddingLayer"),
    num_embeddings_(num_embeddings),
    embedding_dims_(embedding_dims),
    embedding_op_(std::make_unique<EmbeddingOp<T>>(num_embeddings_, embedding_dims_, epsilon))
{
    // param node
    weight_var_ = std::make_unique<Variable>("weight_var", CppTypeToDataType<T>::Type());
    gamma_var_  = std::make_unique<Variable>("gamma_var", CppTypeToDataType<T>::Type());
    beta_var_   = std::make_unique<Variable>("beta_var", CppTypeToDataType<T>::Type());

    weight_var_->SetShape(Shape({(int)num_embeddings_, (int)embedding_dims_}));
    gamma_var_->SetShape(Shape({(int)embedding_dims_}));
    beta_var_->SetShape(Shape({(int)embedding_dims_}));

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* EmbeddingLayer<T>::operator()(Variable* a)
{
    SetInputs({a});

    Variable* c = (*embedding_op_)(a, weight_var_.get(), gamma_var_.get(), beta_var_.get());

    SetOutputs({c});
    return c;
}

template<typename T>
Variable* EmbeddingLayer<T>::operator()(Variable* input_ids, Variable* token_type_ids, Variable* position_ids)
{
    SetInputs({input_ids, token_type_ids, position_ids});

    Variable* out = (*embedding_add_add_layer_norm_op_)(input_ids,
                                                        token_type_ids,
                                                        position_ids,
                                                        word_embeddings_var_.get(),
                                                        token_type_embeddings_var_.get(),
                                                        position_embeddings_var_.get(),
                                                        gamma_var_.get(),
                                                        beta_var_.get());

    SetOutputs({out});
    return out;
}

template<typename T>
void EmbeddingLayer<T>::LoadParam(const T* word_embeddings_ptr,
                                  const T* token_type_embeddings_ptr,
                                  const T* position_embeddings_ptr,
                                  const T* gamma_ptr,
                                  const T* beta_ptr)
{
    word_embeddings_var_->SetValue((char*)word_embeddings_ptr);

    token_type_embeddings_var_->SetValue((char*)token_type_embeddings_ptr);

    position_embeddings_var_->SetValue((char*)position_embeddings_ptr);

    gamma_var_->SetValue((char*)gamma_ptr);

    beta_var_->SetValue((char*)beta_ptr);
}

template<typename T>
void EmbeddingLayer<T>::LoadParam(const T* weight_ptr, const T* gamma_ptr, const T* beta_ptr)
{
    weight_var_->SetValue((char*)weight_ptr);

    gamma_var_->SetValue((char*)gamma_ptr);

    beta_var_->SetValue((char*)beta_ptr);
}

template class EmbeddingLayer<float>;
template class EmbeddingLayer<_Float16>;
template class EmbeddingLayer<ushort>;

}  // namespace lightinfer