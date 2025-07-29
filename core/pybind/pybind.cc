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
    FC_DECLARE_COMMON_PYBIND11_HANDLES(m);

    // Flag management Python API
    m.def("get_flag_value", [](const std::string& name) {
        std::string value;
        if (flashck::GetFlagValue(name.c_str(), value)) {
            return py::cast(value);
        } else {
            throw std::runtime_error("Flag not found: " + name);
        }
    }, py::arg("name"), "Get flag value as string");

    m.def("set_flag_value", [](const std::string& name, const std::string& value) {
        if (!flashck::SetFlagValue(name.c_str(), value.c_str())) {
            throw std::runtime_error("Failed to set flag: " + name);
        }
    }, py::arg("name"), py::arg("value"), "Set flag value by name (string)");

    m.def("print_all_flags", [](bool writable_only) {
        flashck::PrintAllFlags(writable_only);
    }, py::arg("writable_only") = false, "Print all flags to stdout");

    m.def("print_flag_info", [](const std::string& name) {
        if (!flashck::PrintFlagInfo(name.c_str())) {
            throw std::runtime_error("Flag not found: " + name);
        }
    }, py::arg("name"), "Print detailed info for a flag");


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
