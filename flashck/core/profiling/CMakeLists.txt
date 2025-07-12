add_subdirectory(tile)

file(GLOB ProfilingFile CONFIGURE_DEPENDS "*.cc")
add_library(profiling_engine STATIC ${ProfilingFile})
target_include_directories(profiling_engine PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(profiling_engine PUBLIC utils graph)
