#include "flashck/core/module/layers/gemm_layers/linear_layer.h"

namespace flashck {

template<typename T>
class Linear {
public:
    Linear(int in_dims, int out_dims)
    {
        std::string context_name =
            test_name + "_" + flashck::DataTypeToShortString(flashck::CppTypeToDataType<T>::Type());
        flashck::Context::CreateGlobalContext(context_name, flashck::Mode::Inference);
        auto context_ptr = flashck::Context::GetGlobalInstance();

        auto x = std::make_unique<Variable>("x_var", flashck::CppTypeToDataType<T>::Type());
        x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim(k_runtime)});

        auto      linear_layer = std::make_unique<flashck::LinearLayer<T>>(k_runtime, n_runtime, is_split_k, false);
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

    T* GetOutData()
    {
        return reinterpret_cast<T*>(y_out_ater_->GetValue());
    }

    int GetOutShape() const
    {
        return n_runtime;
    }

private:
};

}  // namespace flashck