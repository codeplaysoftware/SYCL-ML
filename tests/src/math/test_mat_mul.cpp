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

#include "ml/math/mat_mul.hpp"
#include "utils/utils.hpp"

template <class T>
void test_square() {
  std::array<T, 4> m1{1.0, 2.0, 3.0, 4.0};
  std::array<T, 4> m2{-1.0, 1.0, 5.0, -2.0};
  std::array<T, 4> m3;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(2, 2));
    m1_buffer.set_final_data(nullptr);
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(2, 2));
    m2_buffer.set_final_data(nullptr);
    ml::matrix_t<T> out_buffer(cl::sycl::range<2>(2, 2));
    ml::mat_mul(q, m1_buffer, m2_buffer, out_buffer);
    out_buffer.set_final_data(m3.data());
    clear_eigen_device();
  }

  /*
  std::cout << "m1:\n";
  ml::print(m1, 2, 2);
  std::cout << "\nm2:\n";
  ml::print(m2, 2, 2);
  std::cout << "\nm3:\n";
  ml::print(m3, 2, 2);
  */

  assert_vec_almost_eq(m3, {9.0, -3.0, 17.0, -5.0});
}

template <class T>
void test_general() {
  std::array<T, 6> m1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::array<T, 3> m2{-1.0, 5.0, 2.0};
  std::array<T, 2> m3;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(2, 3));
    m1_buffer.set_final_data(nullptr);
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(3, 1));
    m2_buffer.set_final_data(nullptr);
    ml::matrix_t<T> out_buffer(cl::sycl::range<2>(2, 1));
    ml::mat_mul(q, m1_buffer, m2_buffer, out_buffer);
    out_buffer.set_final_data(m3.data());
    clear_eigen_device();
  }

  /*
  std::cout << "m1:\n";
  ml::print(m1, 2, 3);
  std::cout << "\nm2:\n";
  ml::print(m2, 3, 1);
  std::cout << "\nm3:\n";
  ml::print(m3, 2, 1);
  */

  assert_vec_almost_eq(m3, {15.0, 33.0});
}

template <class T>
void test_all() {
  test_square<T>();
  test_general<T>();
}

int main() {
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
