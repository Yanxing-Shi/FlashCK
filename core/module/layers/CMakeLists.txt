# ==============================================================================
# Core Layer Components
# ==============================================================================

# Collect all module source files recursively
file(GLOB_RECURSE LAYER_SOURCES CONFIGURE_DEPENDS
    "*.cc"
    "norm_layers/*.cc"
)

# Create kernel
add_library(layer STATIC ${LAYER_SOURCES})

# Link dependencies
target_link_libraries(layer PUBLIC operation)
