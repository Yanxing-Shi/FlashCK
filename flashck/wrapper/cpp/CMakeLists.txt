# ==============================================================================
# FlashCK C++ Header-Only Wrapper Components
# ==============================================================================

# ==============================================================================
# Modular Components
# ==============================================================================

# LayerNorm cpp wrapper
add_library(layernorm_cpp_wrapper INTERFACE norm/layer_norm.h)

# Link with core libraries when available
target_link_libraries(layernorm_cpp_wrapper INTERFACE layer)

# ==============================================================================
# Installation and Export
# ==============================================================================

# Install headers for development use
install(DIRECTORY norm/
    DESTINATION include/flashck/wrapper/cpp/norm
    FILES_MATCHING PATTERN "*.h"
)
