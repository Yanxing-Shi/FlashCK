# ==============================================================================
# FlashCK Core Components
# ==============================================================================

# Add core subdirectories
add_subdirectory(utils)
add_subdirectory(memory)
add_subdirectory(graph)
add_subdirectory(profiling)
add_subdirectory(module)

# # Create core library target
# add_library(fc_core INTERFACE)

# # Link all core components
# target_link_libraries(fc_core INTERFACE
# fc_utils
# fc_memory
# fc_graph
# fc_module
# fc_profiling
# )

# # Add alias
# add_library(FlashCK::core ALIAS fc_core)
