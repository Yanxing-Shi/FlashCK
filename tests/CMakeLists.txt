# ==============================================================================
# FlashCK Tests
# ==============================================================================

# Add C++ wrapper tests
if(BUILD_WRAPPER AND BUILD_TESTS)
    message(STATUS "Building C++ wrapper tests...")
    add_subdirectory(wrapper_cpp)
endif()

# Add FlashCK core tests
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
    message(STATUS "Building core C++ tests...")
    add_subdirectory(cpp)
endif()

# Add Python tests
if(BUILD_PYTHON_BINDINGS AND BUILD_TESTS)
    message(STATUS "Building Python tests...")

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python")
        add_subdirectory(python)
    endif()
endif()
