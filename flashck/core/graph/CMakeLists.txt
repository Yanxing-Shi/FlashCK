# ==============================================================================
# FlashCK Graph Library
# ==============================================================================

# Define source files
file(GLOB GraphFile CONFIGURE_DEPENDS "*.cc")

# Create memory library
add_library(graph STATIC ${GraphFile})

# Link dependencies
target_link_libraries(graph PUBLIC utils memory)
