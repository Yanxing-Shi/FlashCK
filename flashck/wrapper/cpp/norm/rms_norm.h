#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "flashck/core/graph/common.h"
#include "flashck/core/module/layers/norm_layers/rms_norm_layer.h"
#include "flashck/core/profiling/profiling_engine.h"
#include "flashck/core/utils/dtype.h"

namespace flashck {

/**
 * @brief Execute RMSNorm forward pass
 * @tparam T Data type (float, _Float16, ushort)
 * @param x Input tensor [m, n]
 * @param gamma Scale parameters [n]
 * @param m Batch size
 * @param n Feature dimension
 * @param epsilon Numerical stability parameter (default: 1e-5)
 * @return Output tensor [m, n]
 * @throws std::runtime_error on execution failure
 *
 * Computes: output = gamma * x / sqrt(mean(x^2) + epsilon)
 *
 * RMSNorm (Root Mean Square Layer Normalization) is a simplified variant of LayerNorm
 * that omits the mean centering and bias term, focusing only on the scaling by the
 * root mean square. It's commonly used in transformer architectures like LLaMA.
 */
template<typename T>
T* rms_norm_fwd(T* x, T* gamma, int m, int n, float epsilon = 1e-5f)
{
    // Input validation
    FC_ENFORCE_NOT_NULL(x, "RMSNorm: null input tensor x");
    FC_ENFORCE_NOT_NULL(gamma, "RMSNorm: null scale parameter gamma");

    FC_ENFORCE_EQ(IsGpuPointer(x), true, Unavailable("RMSNorm: input tensor x must be a GPU pointer"));
    FC_ENFORCE_EQ(IsGpuPointer(gamma), true, Unavailable("RMSNorm: scale parameter gamma must be a GPU pointer"));

    FC_ENFORCE_EQ(m > 0 && n > 0, true, Unavailable("RMSNorm: invalid dimensions m = {}, n = {}", m, n));

    try {
        // Create unique context
        std::string context_name = std::string("rms_norm_fwd_") + DataTypeToString(CppTypeToDataType<T>::value) + "_"
                                   + std::to_string(m) + "x" + std::to_string(n);

        Context::CreateGlobalContext(context_name);
        auto context_ptr = Context::GetGlobalInstance();

        // Create input variable
        auto x_var = std::make_unique<Variable>("x_var", CppTypeToDataType<T>::value);
        x_var->SetShape({m, n});

        // Create RMSNorm layer
        auto      rms_norm_layer = std::make_unique<RMSNormLayer<T>>(Shape({DDim(n)}));
        Variable* out            = (*rms_norm_layer)(x_var.get(), epsilon);

        // Build computation graph
        context_ptr->BuildContext();

        // Generate optimized kernels
        ProfilingEngine::GetInstance()->GetGraphCodeGen()->CodeGenAndProfiling(context_ptr->GetModelOps(),
                                                                               context_name);

        // Load parameters and execute
        rms_norm_layer->LoadParam(gamma);
        x_var->SetValue(reinterpret_cast<char*>(x));
        rms_norm_layer->Forward();

        return reinterpret_cast<T*>(out->GetValue());
    }
    catch (const std::exception& e) {
        throw std::runtime_error("RMSNorm execution failed: " + std::string(e.what()));
    }
}

}  // namespace flashck