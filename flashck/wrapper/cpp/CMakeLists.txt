# ==============================================================================
# FlashCK C++ Header-Only Wrapper Components
# ==============================================================================

# ==============================================================================
# Modular Components
# ==============================================================================

# Norm cpp wrapper
add_library(layernorm_cpp_wrapper INTERFACE norm/layer_norm.h)
add_library(rmsnorm_cpp_wrapper INTERFACE norm/rms_norm.h)

# Link with core libraries when available
target_link_libraries(layernorm_cpp_wrapper INTERFACE layer)
target_link_libraries(rmsnorm_cpp_wrapper INTERFACE layer)
