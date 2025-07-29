# ==============================================================================
# Core Utils Library
# ==============================================================================

# Find all source files
file(GLOB_RECURSE UtilsSourceFiles CONFIGURE_DEPENDS "*.cc")

# Create utils library
add_library(utils STATIC ${UtilsSourceFiles})

# create json header only library
add_library(json_config INTERFACE)
target_link_libraries(json_config INTERFACE nlohmann_json::nlohmann_json)

# Link dependencies
target_link_libraries(utils PUBLIC
    glog::glog
    gflags
    jinja2cpp
    fmt::fmt
    hip::device
    sqlite3
    json_config
)
