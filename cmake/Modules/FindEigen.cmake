# Copyright (C) Codeplay Software Limited.

# Try to find the Eigen library and its Tensor module.
#
# If the library is found then the `eigen::eigen` target will be exported with
# the required include directories.
#
# Sets the following variables:
#   eigen_FOUND        - whether the system has Eigen
#   eigen_INCLUDE_DIRS - the Eigen include directory

find_path(EIGEN_INCLUDE_DIR
  NAMES unsupported/Eigen/CXX11/Tensor
  PATH_SUFFIXES eigen3 Eigen3
  DOC "The Eigen SYCL Tensor module"
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen
  FOUND_VAR Eigen_FOUND
  REQUIRED_VARS EIGEN_INCLUDE_DIR
)
mark_as_advanced(Eigen_FOUND EIGEN_INCLUDE_DIRS)
if(Eigen_FOUND)
  set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
endif()

if(Eigen_FOUND AND NOT TARGET Eigen)
  add_library(Eigen INTERFACE)
  set_target_properties(Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}"
  )
endif()
if(Eigen_FOUND)
  set(eigen_definitions EIGEN_EXCEPTIONS=1
                        EIGEN_USE_SYCL=1
                        EIGEN_SYCL_USE_DEFAULT_SELECTOR=1)
  find_package(Threads)
  if(Threads_FOUND)
    list(APPEND eigen_definitions EIGEN_SYCL_ASYNC_EXECUTION=1)
    set_property(TARGET Eigen
      APPEND PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads
    )
  endif()
  if(SYCLML_EIGEN_NO_BARRIER)
    list(APPEND eigen_definitions EIGEN_SYCL_DISABLE_ARM_GPU_CACHE_OPTIMISATION=1
                                  EIGEN_SYCL_NO_LOCAL_MEM=1)
  else()
    if(SYCLML_EIGEN_LOCAL_MEM)
      list(APPEND eigen_definitions EIGEN_SYCL_LOCAL_MEM=1)
    endif()
    if(SYCLML_EIGEN_NO_LOCAL_MEM)
      list(APPEND eigen_definitions EIGEN_SYCL_NO_LOCAL_MEM=1)
    endif()
  endif()
  set_target_properties(Eigen PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${eigen_definitions}"
  )
  if(SYCLML_EIGEN_COMPRESS_NAMES)
    set_target_properties(Eigen PROPERTIES
      INTERFACE_COMPUTECPP_FLAGS "-sycl-compress-name"
    )
  endif()
endif()
