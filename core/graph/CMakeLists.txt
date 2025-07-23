# ==============================================================================
# Core Graph Library
# ==============================================================================

file(GLOB GraphFile CONFIGURE_DEPENDS "*.cc")

add_library(graph STATIC ${GraphFile})

target_link_libraries(graph PUBLIC utils memory)
