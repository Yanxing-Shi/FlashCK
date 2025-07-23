# ==============================================================================
# Core Memory Library
# ==============================================================================

# Define source files
set(MemoryFile
    allocator.cc
    memory_manager.cc
)

# Create memory library
add_library(memory STATIC ${MemoryFile})

# Link dependencies
target_link_libraries(memory PUBLIC utils)
