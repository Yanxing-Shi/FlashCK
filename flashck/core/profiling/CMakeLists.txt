add_subdirectory(tile)

file(GLOB ProfilingFile CONFIGURE_DEPENDS "*.cc")
add_library(profiling_engine STATIC ${ProfilingFile})
set_property(TARGET profiling_engine PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(profiling_engine PUBLIC utils graph)
