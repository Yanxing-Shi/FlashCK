cmake_minimum_required(VERSION 3.8)

project(lightinfer LANGUAGES CXX)

# package
## rocm
find_package(ROCM REQUIRED PATHS /opt/rocm)
find_package(hiprand REQUIRED)
find_package(rccl REQUIRED)
find_package(Perl REQUIRED)
# On ROCm rocThrust requires rocPRIM
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
# On ROCm rocPRIM requires hipCUB
find_package(hipcub REQUIRED CONFIG PATHS "/opt/rocm/hipcub")
# "/opt/rocm" - default install prefix
find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")


## HIP
find_package(HIP REQUIRED)
message(STATUS "Build with HIP ${HIP_VERSION}")
link_libraries(hip::device)
add_compile_definitions(__HIP_PLATFORM_HCC__=1)
## glog
find_package(glog REQUIRED)
add_compile_definitions(-DGLOG_USE_GLOG_EXPORT)
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

find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

option(BUILD_MULTI_GPU "Build project about multi-GPU" OFF)

if(BUILD_MULTI_GPU)
  message(STATUS "Add DBUILD_MULTI_GPU, requires MPI and RCCL")
  add_definitions("-DBUILD_MULTI_GPU")
  find_package(MPI REQUIRED)
  find_package(RCCL REQUIRED)
endif()


enable_language(HIP)
set(CMAKE_HIP_STANDARD 23)
set(CMAKE_HIP_EXTENSIONS ON)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# complier
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
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

if (BUILD_MULTI_GPU)
  list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH})
  list(APPEND COMMON_LIB_DIRS /opt/ompi/lib/)
endif()

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

# Add sub directory
add_subdirectory(lightinfer)
# add_subdirectory(examples)
add_subdirectory(tests)

