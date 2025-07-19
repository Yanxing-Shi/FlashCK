# ==============================================================================
# FlashCK Tests
# ==============================================================================

# Add FlashCK core tests
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
    message(STATUS "Building core C++ tests...")
    add_subdirectory(cpp)
endif()
