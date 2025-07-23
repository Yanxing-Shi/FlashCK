#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "flashck/core/graph/common.h"
#include "flashck/core/module/layers/norm_layers/layer_norm_layer.h"
#include "flashck/core/profiling/profiling_engine.h"
#include "flashck/core/utils/dtype.h"

namespace flashck {

namespace cpp {

/**
 * @brief Execute LayerNorm forward pass
 * @tparam T Data type (float, _Float16, ushort)
 * @param x Input tensor [m, n]
 * @param gamma Scale parameters [n]
 * @param beta Bias parameters [n]
 * @param m Batch size
 * @param n Feature dimension
 * @param epsilon Numerical stability parameter (default: 1e-5)
 * @return Output tensor [m, n]
 *
 * Computes: output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
 */
template<typename T>
T* layer_norm_fwd(T* x, T* gamma, T* beta, int m, int n, float epsilon = 1e-5f)
{
    // Input validation
    FC_ENFORCE_NOT_NULL(x, "LayerNorm: null input tensor x");
    FC_ENFORCE_NOT_NULL(gamma, "LayerNorm: null scale parameter gamma");
    FC_ENFORCE_NOT_NULL(beta, "LayerNorm: null bias parameter beta");

    FC_ENFORCE_EQ(IsGpuPointer(x), true, Unavailable("LayerNorm: input tensor x must be a GPU pointer"));
    FC_ENFORCE_EQ(IsGpuPointer(gamma), true, Unavailable("LayerNorm: scale parameter gamma must be a GPU pointer"));
    FC_ENFORCE_EQ(IsGpuPointer(beta), true, Unavailable("LayerNorm: bias parameter beta must be a GPU pointer"));

    FC_ENFORCE_EQ(m > 0 && n > 0, true, Unavailable("LayerNorm: invalid dimensions m = {}, n = {}", m, n));

    try {
        // Create unique context
        std::string context_name = std::string("layer_norm_fwd_") + DataTypeToString(CppTypeToDataType<T>::value) + "_"
                                   + std::to_string(m) + "x" + std::to_string(n);

        Context::CreateGlobalContext(context_name);
        auto context_ptr = Context::GetGlobalInstance();

        // Create input variable
        auto x_var = std::make_unique<Variable>("x_var", CppTypeToDataType<T>::value);
        x_var->SetShape({m, n});

        // Create LayerNorm layer
        auto      layer_norm_layer = std::make_unique<LayerNormLayer<T>>(Shape({DDim(n)}));
        Variable* out              = (*layer_norm_layer)(x_var.get(), epsilon);

        // Build computation graph
        context_ptr->BuildContext();

        // Generate optimized kernels
        ProfilingEngine::GetInstance()->GetGraphCodeGen()->CodeGenAndProfiling(context_ptr->GetModelOps(),
                                                                               context_name);

        // Load parameters and execute
        layer_norm_layer->LoadParam(gamma, beta);
        x_var->SetValue(reinterpret_cast<char*>(x));
        layer_norm_layer->Forward();

        return reinterpret_cast<T*>(out->GetValue());
    }
    catch (const std::exception& e) {
        throw std::runtime_error("LayerNorm execution failed: " + std::string(e.what()));
    }
}

}  // namespace cpp

}  // namespace flashck