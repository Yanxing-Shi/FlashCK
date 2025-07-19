# ==============================================================================
# LayerNorm Examples
# ==============================================================================

# LayerNorm basic example
add_executable(example_layernorm example_layer_norm.cc)

# Link libraries
target_link_libraries(example_layernorm PRIVATE
    layernorm_cpp_wrapper
)

message(STATUS "LayerNorm example configured: example_layer_norm")
