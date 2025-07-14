cmake_minimum_required(VERSION 3.8)

file(GLOB TileNormFile CONFIGURE_DEPENDS "*.cc")

add_library(tile_norm STATIC ${TileNormFile})
target_include_directories(tile_norm PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(tile_norm PUBLIC utils)