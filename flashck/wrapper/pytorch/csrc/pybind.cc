#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "flashck/wrapper/cpp/norm/layer_norm.h"

namespace py = pybind11;

namespace flashck {

at::Tensor layer_norm_fwd(
    at::Tensor input, const std::vector<int64_t>& normalized_shape, at::Tensor gamma, at::Tensor beta, float eps)
{
    // Check the input tensor is not empty
    TORCH_CHECK(!input.numel() == 0, "Input tensor is empty");
    // Check the gamma and beta tensors are not empty
    TORCH_CHECK(!gamma.numel() == 0, "gamma tensor is empty");
    TORCH_CHECK(!beta.numel() == 0, "beta tensor is empty");
    // Check the normalized_shape is not empty
    TORCH_CHECK(!normalized_shape.empty(), "normalized_shape is empty");

    for (int i = 0; i < normalized_shape.size(); ++i) {
        TORCH_CHECK(normalized_shape[i] > 0, "normalized_shape must be greater than 0");
        int input_dim = input.dim() - normalized_shape.size() + i;
        TORCH_CHECK(input.size(input_dim) == normalized_shape[i],
                    "Input tensor size at dimension ",
                    input_dim,
                    " is ",
                    input.size(input_dim),
                    " but expected ",
                    normalized_shape[i]);
    }

    // If the input is a 2D tensor or higher, reshape it to 2D according to the normalized_shape
    std::vector<int64_t> original_shape(input.sizes().begin(), input.sizes().end());
    at::Tensor           input_reshaped = input;

    if (input.dim() != 2) {
        input_reshaped = input.view({input.size(0), -1});
    }

    int m = input_reshaped.size(0);
    int n = input_reshaped.size(1);

    at::Tensor output;

    // Handle different data types
    if (input.dtype() == at::kFloat) {
        auto output_ptr = layer_norm_fwd<float>(
            input_reshaped.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), m, n, eps);
        output = at::from_blob(output_ptr, {m, n}, at::kFloat).clone();
    }
    else if (input.dtype() == at::kHalf) {
        auto output_ptr = layer_norm_fwd<at::Half>(
            input_reshaped.data_ptr<at::Half>(), gamma.data_ptr<at::Half>(), beta.data_ptr<at::Half>(), m, n, eps);
        output = at::from_blob(output_ptr, {m, n}, at::kHalf).clone();
    }
    else if (input.dtype() == at::kBFloat16) {
        auto output_ptr = layer_norm_fwd<at::BFloat16>(input_reshaped.data_ptr<at::BFloat16>(),
                                                       gamma.data_ptr<at::BFloat16>(),
                                                       beta.data_ptr<at::BFloat16>(),
                                                       m,
                                                       n,
                                                       eps);
        output          = at::from_blob(output_ptr, {m, n}, at::kBFloat16).clone();
    }
    else {
        TORCH_CHECK(false, "Unsupported data type for layer norm");
    }

    // Reshape back to original shape if needed
    if (input.dim() != 2) {
        output = output.view(original_shape);
    }

    return output;
}

}  // namespace flashck

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "FlashCK PyTorch Extension";

    // Declare common pybind11 handles
    // FC_DECLARE_COMMON_PYBIND11_HANDLES(m);

    // Layer normalization functions
    m.def("layer_norm_fwd",
          &flashck::layer_norm_fwd,
          "Forward pass of layer normalization",
          py::arg("input"),
          py::arg("normalized_shape"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("eps") = 1e-5);
}
