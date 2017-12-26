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
  if(NOT EXISTS ${MNIST_FULL_PATH}/${FILENAME}.gz)
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

# Need to define _USE_MATH_DEFINES for M_PI when building on MSVC
if(MSVC)
  add_definitions(-D_USE_MATH_DEFINES)
  set(COMPUTECPP_USER_FLAGS ${COMPUTECPP_USER_FLAGS} "-D_USE_MATH_DEFINES")
endif()

# Build each example
file(GLOB_RECURSE SOURCES "src/*.cpp")
foreach(SOURCE ${SOURCES})
  process_source_name(${SOURCE})
  add_executable(${TARGET_NAME} ${SOURCE})
  set_property(TARGET ${TARGET_NAME} PROPERTY COMPUTECPP_INCLUDE_AFTER 1)
  add_sycl_to_target(${TARGET_NAME} ${OUTPUT_SYCL_DIR} ${SOURCE})
endforeach()
