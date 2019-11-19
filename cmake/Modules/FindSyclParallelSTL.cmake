# Copyright (C) Codeplay Software Limited.

# Try to find the SyclParallelSTL library and its Tensor module.
#
# If the library is found then the `SyclParallelSTL` target will be exported with
# the required include directories.
#
# Sets the following variables:
#   SyclParallelSTL_FOUND        - whether the system has SyclParallelSTL
#   SyclParallelSTL_INCLUDE_DIRS - the SyclParallelSTL include directory

find_path(SYCL_PARALLEL_STL_INCLUDE_DIR
  NAMES sycl/execution_policy
  PATH_SUFFIXES include
  DOC "The SyclParallelSTL include folder"
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SyclParallelSTL
  FOUND_VAR SyclParallelSTL_FOUND
  REQUIRED_VARS SYCL_PARALLEL_STL_INCLUDE_DIR
)
mark_as_advanced(SyclParallelSTL_FOUND SYCL_PARALLEL_STL_INCLUDE_DIRS)
if(SyclParallelSTL_FOUND)
  set(SYCL_PARALLEL_STL_INCLUDE_DIRS ${SYCL_PARALLEL_STL_INCLUDE_DIR})
endif()

if(SyclParallelSTL_FOUND AND NOT TARGET SyclParallelSTL)
  add_library(SyclParallelSTL INTERFACE)
  set_target_properties(SyclParallelSTL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SYCL_PARALLEL_STL_INCLUDE_DIR}"
  )
endif()
