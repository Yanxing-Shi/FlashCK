# ==============================================================================
# Core Kernel Components
# ==============================================================================

# Collect all module source files recursively
file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS
    "*.cc"
    "norm_kernels/*.cc"
)

# Create kernel
add_library(kernel STATIC ${KERNEL_SOURCES})

# Link dependencies
target_link_libraries(kernel PUBLIC utils)
