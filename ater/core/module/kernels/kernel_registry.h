#pragma once

#include "ater/core/module/kernels/kernel_factory.h"

#define SOURCE_TYPE(arg_) ater::SourceType::arg_
#define DATA_LAYOUT(arg_) ater::DataLayout::arg_
#define DATATYPE(arg_) ater::DataType::arg_

#define ATER_NARGS(...) _ATER_NARGS((__VA_ARGS__, _ATER_RESQ_N()))
#define _ATER_NARGS(...) _ATER_ARG_N(__VA_ARGS__)
#define _ATER_ARG_N_EXPAND(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
#define _ATER_ARG_N(args) _ATER_ARG_N_EXPAND args
#define _ATER_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define ATER_ID __COUNTER__

#define ATER_CONCAT(arg1, arg2) arg1##arg2

#define ATER_EXPAND(arg) arg

namespace ater {
// kernel register example
// ATER_REGISTER_KERNEL(CK, gemm, all_layout, meta_kernel_class, fp16, fp32)
template<typename T>
class KernelRegister {
public:
    KernelRegister(SourceType source_type, std::string kernel_name, DataLayout layout, DataType dtype)
    {
        KernelKey kernel_key(source_type, layout, dtype);
        KernelFactory::Instance().GetKernelsMap()[kernel_name][kernel_key] = &KernelRegister<T>::Create;
    }

    inline static Kernel* Create()
    {
        return new T;
    }
};

}  // namespace ater

#define ATER_REGISTER_KERNEL(source_type, kernel_name, meta_kernel_class, layout, ...)                                 \
    ATER_KERNEL_REGISTRAR_INIT(source_type, kernel_name, meta_kernel_class, layout, __VA_ARGS__)

#define ATER_KERNEL_REGISTRAR_INIT(source_type, kernel_name, meta_kernel_class, layout, ...)                           \
    ATER_EXPAND(_ATER_KERNEL_REGISTRAR_INIT(                                                                           \
        ATER_NARGS(__VA_ARGS__), source_type, kernel_name, meta_kernel_class, layout, __VA_ARGS__))

#define _ATER_KERNEL_REGISTRAR_INIT(N, source_type, kernel_name, meta_kernel_class, layout, ...)                       \
    ATER_EXPAND(ATER_CONCAT(ATER_KERNEL_REGISTRAR_INIT_,                                                               \
                            N)(source_type, kernel_name, meta_kernel_class, layout, ATER_ID, __VA_ARGS__))

#define _ATER_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)    \
    static const ::ater::KernelRegister<meta_kernel_class> _reg_ater_kernel_##kernel_name##layout##_##registrer_id(    \
        SOURCE_TYPE(source_type), #kernel_name, DATA_LAYOUT(layout), ::ater::CppTypeToDataType<cpp_dtype>::Type());

#define ATER_KERNEL_REGISTRAR_INIT_1(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)     \
    _ATER_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)

#define ATER_KERNEL_REGISTRAR_INIT_2(                                                                                  \
    source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype, ...)                                 \
    _ATER_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)        \
    ATER_EXPAND(ATER_KERNEL_REGISTRAR_INIT_1(source_type, kernel_name, meta_kernel_class, layout, ATER_ID, __VA_ARGS__))

#define ATER_KERNEL_REGISTRAR_INIT_3(                                                                                  \
    source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype, ...)                                 \
    _ATER_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)        \
    ATER_EXPAND(ATER_KERNEL_REGISTRAR_INIT_2(source_type, kernel_name, meta_kernel_class, layout, ATER_ID, __VA_ARGS__))

#define ATER_KERNEL_REGISTRAR_INIT_4(                                                                                  \
    source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype, ...)                                 \
    _ATER_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)        \
    ATER_EXPAND(ATER_KERNEL_REGISTRAR_INIT_3(source_type, kernel_name, meta_kernel_class, layout, ATER_ID, __VA_ARGS__))
