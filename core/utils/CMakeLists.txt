# ==============================================================================
# Core Utils Library
# ==============================================================================

# Find all source files
file(GLOB_RECURSE UtilsSourceFiles CONFIGURE_DEPENDS "*.cc")

# Create utils library
add_library(utils STATIC ${UtilsSourceFiles})

# Link dependencies
target_link_libraries(utils PUBLIC
    glog::glog
    gflags
    jinja2cpp
    fmt::fmt
    hip::device
    sqlite3
)
