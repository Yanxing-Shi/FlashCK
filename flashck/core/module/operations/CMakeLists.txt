# ==============================================================================
# FlashCK Operation Components
# ==============================================================================

# Collect all module source files recursively
file(GLOB_RECURSE OPERATION_SOURCES CONFIGURE_DEPENDS
    "*.cc"
    "norm_ops/*.cc"
)

# Create kernel
add_library(operation STATIC ${OPERATION_SOURCES})

# Link dependencies
target_link_libraries(operation PUBLIC utils profiling kernel)
