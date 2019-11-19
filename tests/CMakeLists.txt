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

include_directories("src")

# Build each test
file(GLOB_RECURSE SOURCES "src/*.cpp")
foreach(SOURCE ${SOURCES})
  add_sycl_ml_executable(${SOURCE})
  if(SYCLML_TEST_DOUBLE)
    target_compile_definitions(${TARGET_NAME} PUBLIC SYCLML_TEST_DOUBLE)
  endif()
  add_test(NAME ${TARGET_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} ${MNIST_RELATIVE_PATH})
  set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "${OUTPUT_SYCL_DIR}")
endforeach()
