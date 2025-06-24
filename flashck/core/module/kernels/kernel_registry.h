#pragma once

#include "flashck/core/module/kernels/kernel_factory.h"

#include "flashck/core/profiler/embedding_operation.h"
#include "flashck/core/utils/string_utils.h"

#define SOURCE_TYPE(arg_) flashck::SourceType::arg_
#define DATA_LAYOUT(arg_) flashck::DataLayout::arg_
#define DATATYPE(arg_) flashck::DataType::arg_

#define LI_NARGS(...) _LI_NARGS((__VA_ARGS__, _LI_RESQ_N()))
#define _LI_NARGS(...) _LI_ARG_N(__VA_ARGS__)
#define _LI_ARG_N_EXPAND(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
#define _LI_ARG_N(args) _LI_ARG_N_EXPAND args
#define _LI_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define LI_ID __COUNTER__

#define LI_CONCAT(arg1, arg2) arg1##arg2

#define LI_EXPAND(arg) arg

namespace flashck {
// kernel register example
// flashck_REGISTER_KERNEL(CK, gemm, all_layout, meta_kernel_class, fp16, fp32)
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

}  // namespace flashck

#define flashck_REGISTER_KERNEL(source_type, kernel_name, meta_kernel_class, layout, ...)                              \
    LI_KERNEL_REGISTRAR_INIT(source_type, kernel_name, meta_kernel_class, layout, __VA_ARGS__)

#define LI_KERNEL_REGISTRAR_INIT(source_type, kernel_name, meta_kernel_class, layout, ...)                             \
    LI_EXPAND(_LI_KERNEL_REGISTRAR_INIT(                                                                               \
        LI_NARGS(__VA_ARGS__), source_type, kernel_name, meta_kernel_class, layout, __VA_ARGS__))

#define _LI_KERNEL_REGISTRAR_INIT(N, source_type, kernel_name, meta_kernel_class, layout, ...)                         \
    LI_EXPAND(LI_CONCAT(LI_KERNEL_REGISTRAR_INIT_,                                                                     \
                        N)(source_type, kernel_name, meta_kernel_class, layout, LI_ID, __VA_ARGS__))

#define _LI_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)      \
    static const ::flashck::KernelRegister<meta_kernel_class> _reg_LI_kernel_##kernel_name##layout##_##registrer_id(   \
        SOURCE_TYPE(source_type), #kernel_name, DATA_LAYOUT(layout), ::flashck::CppTypeToDataType<cpp_dtype>::Type());

#define LI_KERNEL_REGISTRAR_INIT_1(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)       \
    _LI_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)

#define LI_KERNEL_REGISTRAR_INIT_2(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype, ...)  \
    _LI_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)          \
    LI_EXPAND(LI_KERNEL_REGISTRAR_INIT_1(source_type, kernel_name, meta_kernel_class, layout, LI_ID, __VA_ARGS__))

#define LI_KERNEL_REGISTRAR_INIT_3(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype, ...)  \
    _LI_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)          \
    LI_EXPAND(LI_KERNEL_REGISTRAR_INIT_2(source_type, kernel_name, meta_kernel_class, layout, LI_ID, __VA_ARGS__))

#define LI_KERNEL_REGISTRAR_INIT_4(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype, ...)  \
    _LI_CREATE_REGISTRAR_OBJECT(source_type, kernel_name, meta_kernel_class, layout, registrer_id, cpp_dtype)          \
    LI_EXPAND(LI_KERNEL_REGISTRAR_INIT_3(source_type, kernel_name, meta_kernel_class, layout, LI_ID, __VA_ARGS__))

#define LI_REGISTER_KERNEL_FOR_ALL_DTYPE(source_type, kernel_name, meta_kernel_class, layout)                          \
    static const ::flashck::KernelRegister<meta_kernel_class> _reg_LI_kernel_##kernel_name##layout(                    \
        SOURCE_TYPE(source_type), #kernel_name, DATA_LAYOUT(layout), ::flashck::DataType::ALL_DTYPE);

#define LI_REGISTER_KERNEL_FOR_ALL_LAYOUT_DTYPE(source_type, kernel_name, meta_kernel_class)                           \
    static const ::flashck::KernelRegister<meta_kernel_class> _reg_LI_kernel_##kernel_name(                            \
        SOURCE_TYPE(source_type), #kernel_name, ::flashck::DataLayout::ALL_LAYOUT, ::flashck::DataType::ALL_DTYPE);