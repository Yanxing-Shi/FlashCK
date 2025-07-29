#include <pybind11/pybind11.h>

#define FC_DECLARE_COMMON_PYBIND11_HANDLES(m)                                                                          \
    pybind11::enum_<flashck::DataType>(m, "DataType", pybind11::module_local())                                        \
        .value("bool", flashck::DataType::BOOL)                                                                        \
        .value("float8", flashck::DataType::FLOAT8)                                                                    \
        .value("bfloat8", flashck::DataType::BFLOAT8)                                                                  \
        .value("float16", flashck::DataType::FLOAT16)                                                                  \
        .value("bfloat16", flashck::DataType::BFLOAT16)                                                                \
        .value("uint32", flashck::DataType::UINT32)                                                                    \
        .value("int32", flashck::DataType::INT32)                                                                      \
        .value("float32", flashck::DataType::FLOAT32)                                                                  \
        .value("uint64", flashck::DataType::UINT64)                                                                    \
        .value("int64", flashck::DataType::INT64)                                                                      \
        .value("float64", flashck::DataType::FLOAT64);                                                                 \
                                                                                                                       \
    pybind11::enum_<flashck::NormBiasEnum>(m, "NormBiasEnum", pybind11::module_local())                                \
        .value("no_bias", flashck::NormBiasEnum::NO_BIAS)                                                              \
        .value("add_bias", flashck::NormBiasEnum::ADD_BIAS);                                                           \
                                                                                                                       \
    pybind11::enum_<flashck::FusedAddEnum>(m, "FusedAddEnum", pybind11::module_local())                                \
        .value("no_add", flashck::FusedAddEnum::NO_ADD)                                                                \
        .value("pre_add_store", flashck::FusedAddEnum::PRE_ADD_STORE)                                                  \
        .value("pre_add", flashck::FusedAddEnum::PRE_ADD);                                                             \
                                                                                                                       \
    pybind11::enum_<flashck::FusedQuantEnum>(m, "FusedQuantEnum", pybind11::module_local())                            \
        .value("no_sweep", flashck::FusedQuantEnum::NO_SWEEP)                                                          \
        .value("smooth_dynamic_quant", flashck::FusedQuantEnum::SMOOTH_DYNAMIC_QUANT)                                  \
        .value("dynamic_quant", flashck::FusedQuantEnum::DYNAMIC_QUANT);
    
    
