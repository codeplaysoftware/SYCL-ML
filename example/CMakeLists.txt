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

# Download and extract MNIST
set(MNIST_FILES "train-images-idx3-ubyte"
                "train-labels-idx1-ubyte"
                "t10k-images-idx3-ubyte"
                "t10k-labels-idx1-ubyte")
set(MNIST_RELATIVE_PATH "data/mnist/")
set(MNIST_FULL_PATH ${CMAKE_CURRENT_BINARY_DIR}/${MNIST_RELATIVE_PATH})
file(MAKE_DIRECTORY ${MNIST_FULL_PATH})
foreach(FILENAME ${MNIST_FILES})
  if(NOT EXISTS ${MNIST_FULL_PATH}/${FILENAME})
    set(FILE "${FILENAME}.gz")
    set(DL_LINK "http://yann.lecun.com/exdb/mnist/${FILE}")
    file(DOWNLOAD ${DL_LINK} ${MNIST_FULL_PATH}/${FILE} SHOW_PROGRESS)
    if(UNIX)
      execute_process(COMMAND gunzip ${FILE}
                      WORKING_DIRECTORY ${MNIST_FULL_PATH}
                      RESULT_VARIABLE PROC_RET)
      if(${PROC_RET})
        message(FATAL_ERROR "Could not extract ${FILE}")
      endif()
    else()
      message(WARNING "Please, extract the file ${MNIST_FULL_PATH}/${FILE}")
    endif()
  endif()
endforeach()

# Build each example
file(GLOB_RECURSE SOURCES "src/*.cpp")
foreach(SOURCE ${SOURCES})
  add_sycl_ml_executable(${SOURCE})
endforeach()
