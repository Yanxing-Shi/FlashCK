# ==============================================================================
# FlashCK Wrapper Components
# ==============================================================================

# Add C++ wrapper (always build if wrapper is enabled)
add_subdirectory(cpp)

# Add PyTorch wrapper if Python bindings are enabled
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(pytorch)
endif()
