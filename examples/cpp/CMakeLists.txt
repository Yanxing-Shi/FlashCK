# ==============================================================================
# FlashCK C++ Examples
# ==============================================================================

cmake_minimum_required(VERSION 3.21)

# Add LayerNorm examples
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/norm")
    message(STATUS "Adding LayerNorm examples...")
    add_subdirectory(norm)
endif()

# Future examples can be added here:
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gemm")
# add_subdirectory(gemm)
# endif()
#
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fmha")
# add_subdirectory(fmha)
# endif()
message(STATUS "C++ examples configuration completed")
