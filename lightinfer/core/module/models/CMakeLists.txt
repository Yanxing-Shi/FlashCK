cmake_minimum_required(VERSION 3.8)

add_subdirectory(bert)

add_library(feed_data feed_data.cc)
set_property(TARGET feed_data PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(feed_data PUBLIC enforce printf log)

add_library(feed_data_map feed_data_map.cc)
set_property(TARGET feed_data_map PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(feed_data_map PUBLIC feed_data)






