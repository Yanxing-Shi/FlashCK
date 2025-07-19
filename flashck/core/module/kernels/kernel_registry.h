#pragma once

#include "flashck/core/module/kernels/kernel_factory.h"

/// @brief Convenient macros for enum values
#define SOURCE_TYPE(arg_) flashck::SourceType::arg_
#define DATA_LAYOUT(arg_) flashck::DataLayout::arg_
#define DATATYPE(arg_) flashck::DataType::arg_

/// @brief Data type aliases for convenience
#define FP16 FLOAT16
#define FP32 FLOAT32
#define FP64 FLOAT64
#define BF16 BFLOAT16
#define BF8 BFLOAT8
#define FP8 FLOAT8
#define I32 INT32
#define U32 UINT32
#define I64 INT64
#define U64 UINT64

/// @brief Utility macros for variadic argument handling
#define FC_NARGS(...) _FC_NARGS((__VA_ARGS__, _FC_RESQ_N()))
#define _FC_NARGS(...) _FC_ARG_N(__VA_ARGS__)
#define _FC_ARG_N_EXPAND(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
#define _FC_ARG_N(args) _FC_ARG_N_EXPAND args
#define _FC_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define FC_ID __COUNTER__
#define FC_CONCAT(arg1, arg2) arg1##arg2
#define FC_EXPAND(arg) arg

namespace flashck {

/**
 * @brief Auto-registration mechanism for kernel implementations
 * @tparam T Kernel implementation class (must inherit from Kernel)
 *
 * This class provides automatic registration of kernel implementations
 * with the KernelFactory during static initialization.
 */
template<typename T>
class KernelRegister {
public:
    /// @brief Constructor registers kernel with factory
    /// @param source_type Source type of the kernel
    /// @param kernel_name Name identifier for the kernel
    /// @param layout Data layout requirement
    /// @param dtype Data type requirement
    KernelRegister(SourceType source_type, const std::string& kernel_name, DataLayout layout, DataType dtype)
    {
        KernelKey kernel_key(source_type, layout, dtype);
        KernelFactory::Instance().GetKernelsMap()[kernel_name][kernel_key] = &KernelRegister<T>::Create;
    }

    /// @brief Factory method to create kernel instance
    /// @return New instance of kernel implementation
    static Kernel* Create()
    {
        return new T;
    }
};

}  // namespace flashck

/// @brief Main registration macro for kernel implementations
/// @param source_type Source type (LEGACY, TILE, etc.)
/// @param kernel_name String name for the kernel
/// @param meta_kernel_class Kernel implementation class
/// @param layout Data layout requirement
/// @param ... Variable list of data types to register
#define FC_REGISTER_KERNEL(source_type, kernel_name, meta_kernel_class, layout, ...)                                   \
    FC_KERNEL_REGISTRAR_INIT(source_type, kernel_name, meta_kernel_class, layout, __VA_ARGS__)

/// @brief Internal macro for registration initialization
#define FC_KERNEL_REGISTRAR_INIT(source_type, kernel_name, meta_kernel_class, layout, ...)                             \
    FC_EXPAND(_FC_KERNEL_REGISTRAR_INIT(                                                                               \
        FC_NARGS(__VA_ARGS__), source_type, kernel_name, meta_kernel_class, layout, __VA_ARGS__))

/// @brief Dispatch to appropriate registration macro based on argument count
#define _FC_KERNEL_REGISTRAR_INIT(N, source_type, kernel_name, meta_kernel_class, layout, ...)                         \
    FC_EXPAND(FC_CONCAT(FC_KERNEL_REGISTRAR_INIT_,                                                                     \
                        N)(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))

/// @brief Creates a static registration object for a single data type
#define _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)          \
    static const ::flashck::KernelRegister<meta_kernel_class> _reg_FC_kernel_##kernel_name##layout##_##registrer_id(   \
        SOURCE_TYPE(source_type), #kernel_name, DATA_LAYOUT(layout), DATATYPE(dtype));

/// @brief Registration for 1 data type
#define FC_KERNEL_REGISTRAR_INIT_1(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)           \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)

/// @brief Registration for 2 data types
#define FC_KERNEL_REGISTRAR_INIT_2(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype, ...)      \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)              \
    FC_EXPAND(FC_KERNEL_REGISTRAR_INIT_1(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))

/// @brief Registration for 3 data types
#define FC_KERNEL_REGISTRAR_INIT_3(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype, ...)      \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)              \
    FC_EXPAND(FC_KERNEL_REGISTRAR_INIT_2(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))

/// @brief Registration for 4 data types
#define FC_KERNEL_REGISTRAR_INIT_4(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype, ...)      \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)              \
    FC_EXPAND(FC_KERNEL_REGISTRAR_INIT_3(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))

/// @brief Registration for 5 data types
#define FC_KERNEL_REGISTRAR_INIT_5(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype, ...)      \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)              \
    FC_EXPAND(FC_KERNEL_REGISTRAR_INIT_4(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))

/// @brief Registration for 6 data types
#define FC_KERNEL_REGISTRAR_INIT_6(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype, ...)      \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)              \
    FC_EXPAND(FC_KERNEL_REGISTRAR_INIT_5(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))

/// @brief Registration for 7 data types
#define FC_KERNEL_REGISTRAR_INIT_7(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype, ...)      \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)              \
    FC_EXPAND(FC_KERNEL_REGISTRAR_INIT_6(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))

/// @brief Registration for 8 data types
#define FC_KERNEL_REGISTRAR_INIT_8(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype, ...)      \
    _FC_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, dtype)              \
    FC_EXPAND(FC_KERNEL_REGISTRAR_INIT_7(source_type, kernel_name, meta_kernel_class, layout, FC_ID, __VA_ARGS__))
