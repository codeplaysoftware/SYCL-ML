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

#include "ml/math/cov.hpp"
#include "ml/math/mat_ops.hpp"
#include "utils/utils.hpp"

template <class T>
void test_cov_square() {
  static constexpr ml::data_dim D = ml::LIN;
  std::array<T, 9> host_data{1.0, 4.0, 7.0, 2.0, 0.0, -8.0, 1.0, 2.0, 1.0};

  std::array<T, 9> host_cov;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(), cl::sycl::range<2>(3, 3));
    sycl_data.set_final_data(nullptr);
    ml::vector_t<T> sycl_data_avg{cl::sycl::range<1>(3)};

    ml::avg<D>(q, sycl_data, sycl_data_avg);
    ml::center_data<ml::opp<D>()>(q, sycl_data, sycl_data_avg);

    ml::matrix_t<T> sycl_cov(cl::sycl::range<2>(3, 3));
    ml::cov<D>(q, sycl_data, sycl_cov);
    sycl_cov.set_final_data(host_cov.data());
    clear_eigen_device();
  }

  /*
  std::cout << "host data:\n";
  ml::print(host_data, 3, 3);
  std::cout << "\ncov:\n";
  ml::print(host_cov, 3, 3);
  */

  std::array<T, 9> expected{2.0 / 9.0,   -2.0 / 3.0,  -8.0 / 3.0,
                            host_cov[1], 8.0 / 3.0,   10.0,
                            host_cov[2], host_cov[5], 38.0};
  assert_vec_almost_eq(host_cov, expected);
}

template <class T>
void test_cov_general() {
  static constexpr ml::data_dim D = ml::TR;
  // 3 observations that have 2 variables each (transposed)
  std::array<T, 6> host_data{1.0, 2.0, 3.0, 2.0, 2.0, 11.0};

  std::array<T, 4> host_cov;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(), cl::sycl::range<2>(2, 3));
    ml::vector_t<T> sycl_data_avg(cl::sycl::range<1>(2));

    ml::avg<D>(q, sycl_data, sycl_data_avg);
    ml::center_data<ml::opp<D>()>(q, sycl_data, sycl_data_avg);

    ml::matrix_t<T> sycl_cov(cl::sycl::range<2>(2, 2));
    ml::cov<D>(q, sycl_data, sycl_cov);
    sycl_cov.set_final_data(host_cov.data());
    clear_eigen_device();
  }

  /*
  std::cout << "data:\n";
  ml::print(host_data, 3, 2);
  std::cout << "\ncov:\n";
  ml::print(host_cov, 2, 2);
  */

  std::array<T, 4> expected{2.0 / 3.0, 3.0, host_cov[1], 18.0};
  assert_vec_almost_eq(host_cov, expected);
}

template <class T>
void test_all() {
  test_cov_square<T>();
  test_cov_general<T>();
}

int main(void) {
  try {
    test_all<float>();
#ifdef SYCLML_TEST_DOUBLE
    test_all<double>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
