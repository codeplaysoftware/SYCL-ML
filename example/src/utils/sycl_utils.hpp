/**
 * Copyright (C) Codeplay Software Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
    cgh.single_task<init_first_kernel>([]() {});
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

#endif  // EXAMPLE_SRC_UTILS_SYCL_UTILS_HPP
