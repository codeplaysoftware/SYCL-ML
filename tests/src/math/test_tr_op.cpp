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

#include "ml/math/mat_ops.hpp"
#include "utils/utils.hpp"

template <class T>
void test_lin_tr_inplace_mat_op() {
  static constexpr size_t M = 10;
  static constexpr size_t N = 2;

  std::array<T, M * N> m1;
  std::array<T, M * N> expected_m1;
  std::array<T, N * M> m2;

  srand(time(0));
  static constexpr T MAX = 1E2;
  fill_random(m1, -MAX, MAX);

  // Expect first column unchanged, second multiplied by 2
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      m2[j * M + i] = j + 1;
      expected_m1[i * N + j] = m1[i * N + j] * m2[j * M + i];
    }
  }

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(M, N));
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(N, M));  // TR
    m2_buffer.set_final_data(nullptr);

    ml::mat_inplace_binary_op<ml::LIN, ml::TR>(q, m1_buffer, m2_buffer,
                                               std::multiplies<T>());
    clear_eigen_device();
  }

  assert_vec_almost_eq(m1, expected_m1);
}

template <class T>
void test_tr_lin_inplace_mat_op() {
  static constexpr size_t M = 2;
  static constexpr size_t N = 10;

  std::array<T, M * N> m1;
  std::array<T, M * N> expected_m1;
  std::array<T, N * M> m2;

  srand(time(0));
  static constexpr T MAX = 1E2;
  fill_random(m1, -MAX, MAX);

  // Expect first column unchanged, second multiplied by 2
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      m2[j * M + i] = i + 1;
      expected_m1[i * N + j] = m1[i * N + j] * m2[j * M + i];
    }
  }

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(M, N));  // TR
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(N, M));
    m2_buffer.set_final_data(nullptr);

    ml::mat_inplace_binary_op<ml::TR, ml::LIN>(q, m1_buffer, m2_buffer,
                                               std::multiplies<T>());
    clear_eigen_device();
  }

  assert_vec_almost_eq(m1, expected_m1);
}

template <class T>
void test_all() {
  test_lin_tr_inplace_mat_op<T>();
  test_tr_lin_inplace_mat_op<T>();
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
