# Copyright (C) Codeplay Software Limited.

cmake_minimum_required(VERSION 3.2.2)

if(NOT SYCLML_DOWNLOAD_EIGEN)
  find_package(Eigen)
endif()

if(NOT Eigen_FOUND AND (SYCLML_DOWNLOAD_EIGEN OR SYCLML_DOWNLOAD_MISSING_DEPS))
  include(ExternalProject)
  set(EIGEN_REPO "https://gitlab.com/libeigen/eigen" CACHE STRING
    "Eigen repository to use"
  )
  set(EIGEN_GIT_TAG "d0ae052" CACHE STRING
    "Git tag, branch or commit to use for the Eigen library"
  )
  set(EIGEN_SOURCE_DIR ${PROJECT_BINARY_DIR}/Eigen-src)
  if(NOT TARGET Eigen_download)
    ExternalProject_Add(Eigen_download
      GIT_REPOSITORY    ${EIGEN_REPO}
      GIT_TAG           ${EIGEN_GIT_TAG}
      SOURCE_DIR        ${EIGEN_SOURCE_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND     ""
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
    )
  endif()
  set(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR})
  file(MAKE_DIRECTORY ${EIGEN_INCLUDE_DIR})

  find_package(Eigen)
  add_dependencies(Eigen Eigen_download)
  mark_as_advanced(EIGEN_REPO EIGEN_GIT_TAG)
endif()

if(NOT Eigen_FOUND)
  message(FATAL_ERROR
    "Could not find Eigen, consider setting SYCLML_DOWNLOAD_MISSING_DEPS")
endif()
