#ifndef EXAMPLE_SRC_UTILS_SYCL_UTILS_HPP
#define EXAMPLE_SRC_UTILS_SYCL_UTILS_HPP

#include "ml/utils/common.hpp"

class init_first_kernel;

/**
 * @brief Used to avoid measuring OpenCL initialization overhead
 * @param q
 */
void launch_first_kernel(cl::sycl::queue& q) {
  q.submit([](cl::sycl::handler& cgh) {
    cgh.single_task<init_first_kernel>([](){});
  });
}

/**
 * @brief Initialize device_constants and return the queue.
 * @return the sycl queue
 */
cl::sycl::queue& create_queue() {
  ml::device_constants<>::instance = new ml::device_constants<>();
  auto& q = ml::get_eigen_device().sycl_queue();
  launch_first_kernel(q);
  return q;
}

/**
 * @brief Free the singleton device_constants.
 */
void clear_eigen_device() {
  delete ml::get_device_constants();
}

#endif //EXAMPLE_SRC_UTILS_SYCL_UTILS_HPP
