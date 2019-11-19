# Copyright (C) Codeplay Software Limited.

cmake_minimum_required(VERSION 3.2.2)

if(NOT SYCLML_DOWNLOAD_EIGEN)
  find_package(Eigen)
endif()

if(NOT Eigen_FOUND AND (SYCLML_DOWNLOAD_EIGEN OR SYCLML_DOWNLOAD_MISSING_DEPS))
  include(ExternalProject)
  set(EIGEN_REPO "https://bitbucket.org/codeplaysoftware/eigen" CACHE STRING
    "Eigen repository to use"
  )
  set(EIGEN_HG_TAG "b865e5c" CACHE STRING
    "Hg tag, branch or commit to use for the Eigen library"
  )
  set(EIGEN_SOURCE_DIR ${PROJECT_BINARY_DIR}/Eigen-src)
  if(NOT TARGET Eigen_download)
    ExternalProject_Add(Eigen_download
      HG_REPOSITORY     ${EIGEN_REPO}
      HG_TAG            ${EIGEN_HG_TAG}
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
  mark_as_advanced(EIGEN_REPO EIGEN_HG_TAG)
endif()

if(NOT Eigen_FOUND)
  message(FATAL_ERROR
    "Could not find Eigen, consider setting SYCLML_DOWNLOAD_MISSING_DEPS")
endif()
