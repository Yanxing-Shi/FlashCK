#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include "core/pybind/norm/layer_norm.h"

namespace py = pybind11;

namespace flashck {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "FlashCK PyTorch Extension";

    // Declare common pybind11 handles
    // FC_DECLARE_COMMON_PYBIND11_HANDLES(m);

    // Layer normalization functions
    m.def("layer_norm_fwd",
          &pytorch::layer_norm_fwd,
          "Forward pass of layer normalization",
          py::arg("input"),
          py::arg("normalized_shape"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-5);
}
}  // namespace flashck
