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
#include <random>

#include "ml/math/tri_solve.hpp"
#include "utils/utils.hpp"

template <class T>
void test_tri_solve() {
  std::array<T, 9> host_A{1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0};
  std::array<T, 12> host_B{9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
                           3.0, 2.0, 1.0, 0.0, 1.0, 0.0};

  std::array<T, 12> host_Y;
  std::array<T, 12> host_X;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_A(host_A.data(), cl::sycl::range<2>(3, 3));
    sycl_A.set_final_data(nullptr);
    ml::matrix_t<T> sycl_B(host_B.data(), cl::sycl::range<2>(3, 4));
    sycl_B.set_final_data(nullptr);

    ml::matrix_t<T> sycl_Y{cl::sycl::range<2>(3, 4)};
    ml::matrix_t<T> sycl_X{cl::sycl::range<2>(3, 4)};

    ml::tri_solve<ml::LIN, ml::TR>(q, sycl_Y, sycl_A, sycl_B);
    ml::tri_solve<ml::LIN, ml::LIN>(q, sycl_X, sycl_A, sycl_Y);

    sycl_Y.set_final_data(host_Y.data());
    sycl_X.set_final_data(host_X.data());
    clear_eigen_device();
  }

  /*
  std::cout << "Y:\n";
  ml::print(host_Y, 3, 4);
  std::cout << "\nX:\n";
  ml::print(host_X, 3, 4);
  */

  std::array<T, 12> expected_Y{9.0,   8.0,  7.0,    6.0,  -3.25,    -3.0,
                               -2.75, -2.5, -1.625, -1.5, -1.04167, -0.91667};
  std::array<T, 12> expected_X{10.76042, 9.62500,  8.46181,  7.32639,
                               -0.47396, -0.43750, -0.47049, -0.43403,
                               -0.27083, -0.25000, -0.17361, -0.15278};

  assert_vec_almost_eq(host_Y, expected_Y);
  assert_vec_almost_eq(host_X, expected_X);
}

template <class T>
void test_tri_solve_tr() {
  std::array<T, 9> host_A{1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0};
  std::array<T, 12> host_B{9.0, 5.0, 1.0, 8.0, 4.0, 0.0,
                           7.0, 3.0, 1.0, 6.0, 2.0, 0.0};

  std::array<T, 12> host_Y;
  std::array<T, 12> host_X;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_A(host_A.data(), cl::sycl::range<2>(3, 3));
    sycl_A.set_final_data(nullptr);
    ml::matrix_t<T> sycl_B(host_B.data(), cl::sycl::range<2>(4, 3));
    sycl_B.set_final_data(nullptr);

    ml::matrix_t<T> sycl_Y{cl::sycl::range<2>(4, 3)};
    ml::matrix_t<T> sycl_X{cl::sycl::range<2>(4, 3)};

    ml::tri_solve<ml::TR, ml::TR>(q, sycl_Y, sycl_A, sycl_B);
    ml::tri_solve<ml::TR, ml::LIN>(q, sycl_X, sycl_A, sycl_Y);

    sycl_Y.set_final_data(host_Y.data());
    sycl_X.set_final_data(host_X.data());
    clear_eigen_device();
  }

  /*
  std::cout << "Y:\n";
  ml::print(host_Y, 4, 3);
  std::cout << "\nX:\n";
  ml::print(host_X, 4, 3);
  */

  std::array<T, 12> expected_Y{9.0, -3.25, -1.625,   8.0, -3.0, -1.5,
                               7.0, -2.75, -1.04167, 6.0, -2.5, -0.91667};
  std::array<T, 12> expected_X{10.76042, -0.47396, -0.27083, 9.62500,
                               -0.43750, -0.25000, 8.46181,  -0.47049,
                               -0.17361, 7.32639,  -0.43403, -0.15278};

  assert_vec_almost_eq(host_Y, expected_Y);
  assert_vec_almost_eq(host_X, expected_X);
}

template <class T>
void test_all() {
  test_tri_solve<T>();
  test_tri_solve_tr<T>();
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
