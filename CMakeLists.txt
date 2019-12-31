# Copyright (C) Codeplay Software Limited.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.4.3)
project(SYCL-ML)

if(MSVC)
  message(WARNING "Windows support is only experimental for now")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Configuration options controlling automatic downloading of dependencies.
option(SYCLML_DOWNLOAD_EIGEN "Download Eigen headers" ON)
option(SYCLML_DOWNLOAD_MISSING_DEPS
  "Download any dependencies which cannot be found" ON)

# Eigen configuration options.
option(SYCLML_EIGEN_LOCAL_MEM
  "Only compile the local memory versions of Eigen kernels" ON)
option(SYCLML_EIGEN_NO_LOCAL_MEM
  "Only compile the no local memory versions of Eigen kernels" OFF)
option(SYCLML_EIGEN_COMPRESS_NAMES
  "Compress Eigen SYCL kernel names" OFF)
option(SYCLML_EIGEN_NO_BARRIER
  "Use Eigen matmul which does not use barriers (implies NO_LOCAL_MEM)" OFF)

# ComputeCpp configuration options.
include(FindComputeCpp)
# Avoid namespace ambiguities with older compilers
list(APPEND COMPUTECPP_USER_FLAGS -DCOMPUTECPP_DISABLE_SYCL_NAMESPACE_ALIAS)
option(SYCLML_COMPUTECPP_USE_SERIAL_MEMOP
  "Replace memory operations (eg memset) in kernels with serial operations." OFF)
if(NOT SYCLML_COMPUTECPP_USE_SERIAL_MEMOP)
  list(APPEND COMPUTECPP_USER_FLAGS -no-serial-memop)
endif()
set(COMPUTECPP_DRIVER_DEFAULT_VALUE OFF)
if (NOT MSVC)
  set(COMPUTECPP_DRIVER_DEFAULT_VALUE ON)
endif()
option(SYCLML_COMPUTECPP_USE_COMPILER_DRIVER
  "Use ComputeCpp driver instead of a 2 steps compilation"
  ${COMPUTECPP_DRIVER_DEFAULT_VALUE}
)
if(SYCLML_COMPUTECPP_USE_COMPILER_DRIVER)
  set(CMAKE_CXX_COMPILER ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE})
endif()
option(SYCLML_COMPUTECPP_SPLIT_MODULES
  "Split modules to speed up compilation." ON)
if(NOT SYCLML_COMPUTECPP_SPLIT_MODULES)
  list(APPEND COMPUTECPP_USER_FLAGS -fsycl-split-modules=20)
endif()

# Tests configuration
option(SYCLML_TEST_DOUBLE "Add tests using double type." OFF)

# Set compile options for host and device compilers
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
foreach(flag -Wall -Wextra -Wpedantic)
  check_cxx_compiler_flag(${flag} is_flag_supported)
  if(is_flag_supported)
    add_compile_options(${flag})
  endif()
endforeach()
if(MSVC)
  add_compile_options(/bigobj)
endif()

include(HandleEigen)

# Include SYCL-ML
include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(sycl_ml INTERFACE)
target_include_directories(sycl_ml INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(sycl_ml INTERFACE
  ComputeCpp::ComputeCpp
  Eigen
)

function(add_sycl_ml_executable SOURCE)
  get_filename_component(SOURCE_NAME_WE ${SOURCE} NAME_WE)
  get_filename_component(DIR ${SOURCE} DIRECTORY)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/" OUTPUT_SYCL_DIR ${DIR})
  file(MAKE_DIRECTORY ${OUTPUT_SYCL_DIR})
  set(TARGET_NAME ${SOURCE_NAME_WE})
  set(TARGET_NAME ${TARGET_NAME} PARENT_SCOPE)
  set(OUTPUT_SYCL_DIR ${OUTPUT_SYCL_DIR} PARENT_SCOPE)

  add_executable(${TARGET_NAME} ${SOURCE})
  target_link_libraries(${TARGET_NAME} sycl_ml)
  set_property(TARGET ${TARGET_NAME} PROPERTY COMPUTECPP_INCLUDE_AFTER 1)
  # Workaround INTERFACE_SYSTEM_INCLUDE_DIRECTORIES not being propagated
  # Add Eigen as system include
  target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:Eigen,INTERFACE_INCLUDE_DIRECTORIES>)
  add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${SOURCE})
endfunction()

add_subdirectory(example)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
