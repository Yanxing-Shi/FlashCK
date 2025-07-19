# ==============================================================================
# Norm Examples
# ==============================================================================

# LayerNorm basic example
add_executable(example_layernorm example_layer_norm.cc)

# Link libraries
target_link_libraries(example_layernorm PRIVATE
    layernorm_cpp_wrapper
)

# RMSNorm basic example
add_executable(example_rmsnorm example_rms_norm.cc)

# Link libraries
target_link_libraries(example_rmsnorm PRIVATE
    rmsnorm_cpp_wrapper
)

message(STATUS "LayerNorm example configured: example_layer_norm")
