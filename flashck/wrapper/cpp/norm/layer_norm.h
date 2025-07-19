#pragma once

#include <memory>
#include <vector>

#include "flashck/core/graph/context.h"
#include "flashck/core/module/common.h"
#include "flashck/core/module/layers/norm_layers/layer_norm_layer.h"
#include "flashck/core/profiling/profiling_engine.h"

namespace flashck {

template<typename T>
T* layer_norm_fwd_static(T* x, T* gamma, T* beta, int m, int n, int emb_dims, float epsilon = 1e-5)
{
    std::string context_name = "layer_norm_fwd_static" + "_" + DataTypeToString(flashck::CppTypeToDataType<T>::Type());

    flashck::Context::CreateGlobalContext(context_name);
    auto context_ptr = flashck::Context::GetGlobalInstance();

    auto x_var = std::make_unique<Variable>("x_var", CppTypeToDataType<T>::Type());
    x_var->SetShape({m, n});

    auto layer_norm_layer = std::make_unique<flashck::LayerNormLayer<T>>(flashck::Shape({flashck::DDim(emb_dims)}));
    Variable* out         = (*layer_norm_layer)(x_var.get(), epsilon);
    context_ptr->BuildContext();

    ProfilingEngine::GetInstance()->GetGraphCodeGen()->CodeGenAndProfiling(context_ptr->GetModelOps(), context_name);

    layer_norm_layer->LoadParam(reinterpret_cast<T*>(gamma), reinterpret_cast<T*>(beta));
    x_var->SetShape({m, n});
    x_var->SetValue(x);
    gamma_var->SetValue(gamma);
    beta_var->SetValue(beta);

    layer_norm_layer->Forward();

    return out->GetValue<T>();
}

template<typename T>
T* layer_norm_fwd_dynamic(
    T* x, T* gamma, T* beta, const std::vector<int>& m_range, int m, int n, int emb_dims, float epsilon = 1e-5)
{
    std::string context_name = "layer_norm_fwd_dynamic" + "_" + DataTypeToString(flashck::CppTypeToDataType<T>::Type());

    flashck::Context::CreateGlobalContext(context_name);
    auto context_ptr = flashck::Context::GetGlobalInstance();

    auto x_var = std::make_unique<Variable>("x_var", CppTypeToDataType<T>::Type());
    x_var->SetShape({DDim({m_range[0], m_range[1]}), DDim(n)});

    auto layer_norm_layer = std::make_unique<flashck::LayerNormLayer<T>>(flashck::Shape({flashck::DDim(emb_dims)}));
    Variable* out         = (*layer_norm_layer)(x_var.get(), epsilon);
    context_ptr->BuildContext();

    ProfilingEngine::GetInstance()->GetGraphCodeGen()->CodeGenAndProfiling(context_ptr->GetModelOps(), context_name);

    layer_norm_layer->LoadParam(reinterpret_cast<T*>(gamma), reinterpret_cast<T*>(beta));
    x_var->SetShape({m, n});
    x_var->SetValue(x);
    gamma_var->SetValue(gamma);
    beta_var->SetValue(beta);

    layer_norm_layer->Forward();

    return out->GetValue<T>();
}

}  // namespace flashck