# ==============================================================================
# FlashCK Core and Wrapper Components
# ==============================================================================

# Add core FlashCK library
add_subdirectory(core)

# Add wrapper components
add_subdirectory(wrapper)

# # Create main FlashCK target
# add_library(flashck_main INTERFACE)
# target_link_libraries(flashck_main INTERFACE flashck_core)

# # Add wrapper if enabled
# if(BUILD_WRAPPER)
# target_link_libraries(flashck_main INTERFACE flashck_wrapper)
# endif()

# # Add alias for easier use
# add_library(FlashCK::FlashCK ALIAS flashck_main)

# # Install main target
# install(TARGETS flashck_main
# EXPORT FlashCKTargets
# ARCHIVE DESTINATION lib
# LIBRARY DESTINATION lib
# RUNTIME DESTINATION bin
# )
