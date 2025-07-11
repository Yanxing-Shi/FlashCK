add_library(kernel_factory STATIC kernel_factory.cc)
set_property(TARGET kernel_factory PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(kernel_factory PUBLIC enforce)

# add_subdirectory(gemm_kernels)
# add_subdirectory(embedding_kernels)
add_subdirectory(norm_kernels)

# add_subdirectory(fmha_kernels)
