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
#include <iostream>

#include "ml/math/mat_ops.hpp"
#include "utils/utils.hpp"

template <class T, ml::data_dim D>
void test_center() {
  constexpr auto NB_OBS = 5LU;
  constexpr auto ACT_SIZE_OBS = 3LU;
  std::array<T, NB_OBS * ACT_SIZE_OBS> host_data{1.0,  4.0, 7.0,  2.0,  0.0,
                                                 -8.0, 1.0, 2.0,  1.0,  0.0,
                                                 0.0,  1.0, -5.0, -4.0, -3.0};

  std::array<T, ACT_SIZE_OBS> host_avg_data;
  std::array<T, NB_OBS * ACT_SIZE_OBS> host_center_data;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(),
                              cl::sycl::range<2>(NB_OBS, ACT_SIZE_OBS));
    ml::vector_t<T> sycl_data_avg{cl::sycl::range<1>(ACT_SIZE_OBS)};

    ml::avg<D>(q, sycl_data, sycl_data_avg);
    ml::center_data<ml::opp<D>()>(q, sycl_data, sycl_data_avg);

    sycl_data.set_final_data(host_center_data.data());
    sycl_data_avg.set_final_data(host_avg_data.data());
    clear_eigen_device();
  }

  /*
  std::cout << "host data:\n";
  ml::print(host_data, NB_OBS, ACT_SIZE_OBS);
  std::cout << "\navg data:\n";
  ml::print(host_avg_data, 1, ACT_SIZE_OBS);
  std::cout << "\ncenter data:\n";
  ml::print(host_center_data, NB_OBS, ACT_SIZE_OBS);
  */

  // avg data
  assert_vec_almost_eq(host_avg_data, {-0.2, 0.4, -0.4});

  // center data
  assert_vec_almost_eq(host_center_data,
                       {1.2, 3.6, 7.4, 2.2, -0.4, -7.6, 1.2, 1.6, 1.4, 0.2,
                        -0.4, 1.4, -4.8, -4.4, -2.6});
}

int main() {
  try {
    test_center<float, ml::ROW>();
#ifdef SYCLML_TEST_DOUBLE
    test_center<double, ml::ROW>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
