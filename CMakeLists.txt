cmake_minimum_required(VERSION 3.8)

project(ater LANGUAGES CXX)

option(DEBUG_MODE "debug computation result" OFF)
option(GIT_AUTOCLONE_CK "Check submodules during build" ON)


# package
## rocm
find_package(ROCM REQUIRED PATHS /opt/rocm)
find_package(hiprand REQUIRED)
find_package(rccl REQUIRED)
## HIP
find_package(HIP REQUIRED)
message(STATUS "Build with HIP ${HIP_VERSION}")
link_libraries(hip::device)
add_compile_definitions(__HIP_PLATFORM_HCC__=1)
## glog
find_package(glog REQUIRED)
## gflags
find_package(gflags REQUIRED)
## fmt
find_package(fmt REQUIRED)
## jinjia2cpp
set (JINJA2CPP_DEPS_MODE internal)
find_package(jinja2cpp REQUIRED)
## openssl
find_package(OpenSSL REQUIRED)
## boost
find_package(Boost REQUIRED)
## sqlite3
find_package(SQLite3 REQUIRED)


# complier
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# SET(CMAKE_CXX_COMPILER "ccache")
# SET(CMAKE_CXX_COMPILER_ARG1 "g++")

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O0 -Wall")
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")


# include header
set(CK_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/composable_kernel/include)
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CK_HEADER_DIR}
  ${HIP_INCLUDE_DIRS}
)

message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")
include_directories(
  ${COMMON_HEADER_DIRS}
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add sub directory
add_subdirectory(ater)
# add_subdirectory(examples)
# add_subdirectory(tests)


# # ater library
# add_library(Ater STATIC
#             $<TARGET_OBJECTS:AterGraph>
#             $<TARGET_OBJECTS:Allocator>
#             $<TARGET_OBJECTS:MemoryManager>
#             $<TARGET_OBJECTS:EmbKernel>
#             $<TARGET_OBJECTS:FMHAKernel>
#             $<TARGET_OBJECTS:GemmKernel>
#             $<TARGET_OBJECTS:LayerNormalizeKernel>
#             $<TARGET_OBJECTS:LayerNormalizeOp>
#             $<TARGET_OBJECTS:GemmBiasActOp>
#             $<TARGET_OBJECTS:EncoderEmbOp>
#             $<TARGET_OBJECTS:BatchGemmBiasOp>
#             $<TARGET_OBJECTS:BatchGemmSoftmaxGemmPermuteOp>
#             $<TARGET_OBJECTS:GemmAddBiasLayerNormalizeOp>
#             $<TARGET_OBJECTS:LayerNormalizeLayer>
#             $<TARGET_OBJECTS:EncoderEmbLayer>
#             $<TARGET_OBJECTS:MultiHeadAttentionLayer>
#             $<TARGET_OBJECTS:FeedForwardLayer>
#             $<TARGET_OBJECTS:BertPoolerLayer>
#             $<TARGET_OBJECTS:TransformerEncoderLayer>
#             $<TARGET_OBJECTS:BertWeight>
#             $<TARGET_OBJECTS:Bert>)

# set_target_properties(Ater PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(Ater PROPERTIES HIP_RESOLVE_DEVICE_SYMBOLS ON)
# target_link_libraries(Ater PUBLIC hip::device hip::hiprand spdlog::spdlog)
