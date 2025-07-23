# ==============================================================================
# Core Profiling Components
# ==============================================================================

# Collect all profiling source files recursively
file(GLOB_RECURSE PROFILING_SOURCES CONFIGURE_DEPENDS
    "*.cc"
    "tile/norm/*.cc"
    "tile/fmha/*.cc"
    "legacy/gemm/*.cc"
)

# Create single unified profiling library
add_library(profiling STATIC ${PROFILING_SOURCES})

# Link dependencies
target_link_libraries(profiling PUBLIC
    utils
    graph
)
