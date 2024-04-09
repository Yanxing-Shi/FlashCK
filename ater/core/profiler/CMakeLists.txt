set(AterProfilerFile
    build_cache.cc
    builder.cc
    gemm_gpu_profiler_runner.cc
    gemm_operation.cc
    generator.cc
    emitters.cc
    profile_cache.cc
    target.cc)
add_library(ater_profiler STATIC ${AterProfilerFile})
set_property(TARGET ater_profiler PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ater_profiler PUBLIC OpenSSL::Crypto jinja2cpp enforce SQLite::SQLite3 rocm_info pthread gemm_rcr_op flags)