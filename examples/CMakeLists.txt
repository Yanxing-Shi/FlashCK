# ==============================================================================
# FlashCK Examples
# ==============================================================================

message(STATUS "Configuring FlashCK examples...")

# Add C++ examples
if(BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
    message(STATUS "Building C++ examples...")
    add_subdirectory(cpp)
endif()

message(STATUS "Examples configuration completed")