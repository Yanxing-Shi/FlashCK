add_library(kernel_factory STATIC kernel_factory.cc)
target_include_directories(kernel_factory PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(kernel_factory PUBLIC utils)

# add_subdirectory(gemm_kernels)
# add_subdirectory(embedding_kernels)
add_subdirectory(norm_kernels)

# add_subdirectory(fmha_kernels)
