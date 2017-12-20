include_directories("src")

# Build each test
enable_testing()
file(GLOB_RECURSE SOURCES "src/*.cpp")
foreach(SOURCE ${SOURCES})
  process_source_name(${SOURCE})
  add_executable(${TARGET_NAME} ${SOURCE})
  set_property(TARGET ${TARGET_NAME} PROPERTY COMPUTECPP_INCLUDE_AFTER 1)
  add_sycl_to_target(${TARGET_NAME} ${OUTPUT_SYCL_DIR} ${SOURCE})

  add_test(NAME ${TARGET_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} ${MNIST_RELATIVE_PATH})
  set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "${OUTPUT_SYCL_DIR}")
endforeach()
