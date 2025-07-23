# ==============================================================================
# FlashCK C++ Header-Only Wrapper Components
# ==============================================================================

# ==============================================================================
# Modular Components
# ==============================================================================

add_library(layernorm_cpp_wrapper INTERFACE norm/layer_norm.h)
add_library(rmsnorm_cpp_wrapper INTERFACE norm/rms_norm.h)

target_link_libraries(layernorm_cpp_wrapper INTERFACE layer)
target_link_libraries(rmsnorm_cpp_wrapper INTERFACE layer)

install(
    DIRECTORY norm/
    DESTINATION include/flashck
    FILES_MATCHING PATTERN "*.h"
)
