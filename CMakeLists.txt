cmake_minimum_required(VERSION 3.0)
project(SYCL-ML)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if (MSVC)
  message(WARNING "Windows support is only experimental for now")
endif()

# Include ComputeCpp
if(NOT COMPUTECPP_PACKAGE_ROOT_DIR AND EXISTS $ENV{COMPUTECPP_PACKAGE_ROOT_DIR})
  set(COMPUTECPP_PACKAGE_ROOT_DIR $ENV{COMPUTECPP_PACKAGE_ROOT_DIR})
endif()
set(COMPUTECPP_FORCE_INCLUDE_SYCL_FILE OFF CACHE BOOL "Include SYCL file manually" FORCE)
include(FindComputeCpp)

# Set compile options for host and device compilers
set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release") # Default to release
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()
add_compile_options(-Wall)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wextra -Werror -pedantic)
endif()
if(MSVC)
  add_compile_options(/bigobj)
endif()

execute_process(COMMAND arch OUTPUT_VARIABLE ARCH OUTPUT_QUIET ERROR_QUIET)
if(ARCH MATCHES ".*aarch.*")
  list(APPEND COMPUTECPP_USER_FLAGS -Rno-sycl-serial-memop)
else()
  list(APPEND COMPUTECPP_USER_FLAGS -no-serial-memop)
endif()

# Include Eigen
if(NOT EIGEN_INCLUDE_DIRS AND EXISTS $ENV{EIGEN_INCLUDE_DIRS})
  set(EIGEN_INCLUDE_DIRS $ENV{EIGEN_INCLUDE_DIRS})
endif()
if(EIGEN_INCLUDE_DIRS)
  include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
  list(APPEND COMPUTECPP_USER_FLAGS -isystem ${EIGEN_INCLUDE_DIRS})
  message(STATUS "Found Eigen: " ${EIGEN_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Please set EIGEN_INCLUDE_DIRS")
endif()
add_compile_options(-DDISABLE_SKINNY=1)
list(APPEND COMPUTECPP_USER_FLAGS -DDISABLE_SKINNY=1)

# Include SyclParallelSTL
if(NOT SYCL_PARALLEL_STL_ROOT AND EXISTS $ENV{SYCL_PARALLEL_STL_ROOT})
  set(SYCL_PARALLEL_STL_ROOT $ENV{SYCL_PARALLEL_STL_ROOT})
endif()
if(SYCL_PARALLEL_STL_ROOT)
  include_directories(SYSTEM ${SYCL_PARALLEL_STL_ROOT}/include)
  list(APPEND COMPUTECPP_USER_FLAGS -isystem ${SYCL_PARALLEL_STL_ROOT}/include)
  message(STATUS "Found SYCLParallelSTL: " ${SYCL_PARALLEL_STL_ROOT}/include)
else()
  message(FATAL_ERROR "Please set SYCL_PARALLEL_STL_ROOT")
endif()

# Include SYCL-ML
include_directories(${CMAKE_SOURCE_DIR}/include)

# Get the target name and the directory used to generate the sycl file
function(process_source_name SOURCE)
  get_filename_component(SOURCE_NAME_WE ${SOURCE} NAME_WE)
  get_filename_component(DIR ${SOURCE} DIRECTORY)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/" OUTPUT_SYCL_DIR ${DIR})
  file(MAKE_DIRECTORY ${OUTPUT_SYCL_DIR})
  set(TARGET_NAME ${SOURCE_NAME_WE} PARENT_SCOPE)
  set(OUTPUT_SYCL_DIR ${OUTPUT_SYCL_DIR} PARENT_SCOPE)
endfunction()

# Include all subdirectories with a CMakeLists.txt file
file(GLOB _ALL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_FILE ${_ALL_FILES})
  if(IS_DIRECTORY ${_FILE})
    if(EXISTS ${_FILE}/CMakeLists.txt)
     add_subdirectory(${_FILE})
    endif()
  endif()
endforeach()
