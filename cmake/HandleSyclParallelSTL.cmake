# Copyright (C) Codeplay Software Limited.

cmake_minimum_required(VERSION 3.2.2)

if(NOT SYCLML_DOWNLOAD_SYCL_PARALLEL_STL)
  find_package(SyclParallelSTL)
endif()

if(NOT SyclParallelSTL_FOUND AND (SYCLML_DOWNLOAD_SYCL_PARALLEL_STL OR SYCLML_DOWNLOAD_MISSING_DEPS))
  include(ExternalProject)
  set(SYCL_PARALLEL_STL_REPO "https://github.com/KhronosGroup/SyclParallelSTL" CACHE STRING
    "SyclParallelSTL repository to use"
  )
  set(SYCL_PARALLEL_STL_GIT_TAG "b2cdb35" CACHE STRING
    "Git tag, branch or commit to use for the SyclParallelSTL library"
  )
set(SYCL_PARALLEL_STL_SOURCE_DIR ${PROJECT_BINARY_DIR}/SyclParallelSTL-src)
  if(NOT TARGET SyclParallelSTL_download)
    ExternalProject_Add(SyclParallelSTL_download
      GIT_REPOSITORY    ${SYCL_PARALLEL_STL_REPO}
      GIT_TAG           ${SYCL_PARALLEL_STL_GIT_TAG}
      SOURCE_DIR        ${SYCL_PARALLEL_STL_SOURCE_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND     ""
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
    )
  endif()
  set(SYCL_PARALLEL_STL_INCLUDE_DIR ${SYCL_PARALLEL_STL_SOURCE_DIR}/include)
  file(MAKE_DIRECTORY ${SYCL_PARALLEL_STL_INCLUDE_DIR})

  find_package(SyclParallelSTL)
  add_dependencies(SyclParallelSTL SyclParallelSTL_download)
  mark_as_advanced(SYCL_PARALLEL_STL_REPO SYCL_PARALLEL_STL_GIT_TAG)
endif()

if(NOT SyclParallelSTL_FOUND)
  message(FATAL_ERROR
    "Could not find SyclParallelSTL, consider setting SYCLML_DOWNLOAD_MISSING_DEPS")
endif()
