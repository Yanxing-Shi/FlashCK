# set(ProfilerFile
#     build_cache.cc
#     builder.cc
#     gpu_profiler_runner.cc
#     gemm_operation.cc
#     norm_operation.cc
#     embedding_operation.cc
#     fmha_fwd_operation.cc
#     fmha_fwd_splitkv_operation.cc
#     fmha_fwd_splitkv_combine_operation.cc
#     fmha_fwd_appendkv_operation.cc
#     generator.cc
#     emitters.cc
#     profile_cache.cc
#     target.cc)
set(ProfilerFile
    library.cc)
add_library(flashck_profiler STATIC ${ProfilerFile})
set_property(TARGET flashck_profiler PROPERTY POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(flashck_profiler PUBLIC OpenSSL::Crypto jinja2cpp enforce SQLite::SQLite3 rocm_info pthread flags)


