#include "lightinfer/core/module/layers/gemm_layers/linear_layer.h"

namespace lightinfer {

template<typename T>
class Linear {
    Linear(int in_dims, int out_dims)
    {
        std::string context_name =
            test_name + "_" + lightinfer::DataTypeToShortString(lightinfer::CppTypeToDataType<T>::Type());
        lightinfer::Context::CreateGlobalContext(context_name, lightinfer::Mode::Inference);
        auto context_ptr = lightinfer::Context::GetGlobalInstance();

        auto x = std::make_unique<Variable>("x_var", lightinfer::CppTypeToDataType<T>::Type());
        x->SetShape({lightinfer::DDim({1, m_max[0]}), lightinfer::DDim(k_runtime)});

        auto      linear_layer = std::make_unique<lightinfer::LinearLayer<T>>(k_runtime, n_runtime, is_split_k, false);
        Variable* y_out_ater   = (*linear_layer)(x.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();
    }

    void operator()(T* w, T* x)
    {
        linear_layer->LoadParam(reinterpret_cast<T*>(w));
        x->SetValue((char*)x);
        torch::Tensor y_pt;

        x->SetShape({m_runtime[0], k_runtime});
        y_pt = GetZerosTorchTensor<T>({m_runtime[0], n_runtime});

        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        linear_layer->Forward();
    }
};

}  // namespace lightinfer