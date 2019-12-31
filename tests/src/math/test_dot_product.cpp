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

#include "ml/math/vec_ops.hpp"
#include "utils/utils.hpp"

template <class T>
void test_dot_product_self() {
  constexpr size_t SIZE = 4;
  std::array<T, SIZE> in{1, 0.5, -1, 0};
  T res;

  {
    cl::sycl::queue& q = create_queue();
    ml::vector_t<T> sycl_vec(in.data(), cl::sycl::range<1>(in.size()));
    res = ml::sycl_dot_product(q, sycl_vec);
    clear_eigen_device();
  }

  /*
  for (unsigned i = 0; i < SIZE; ++i) {
    std::cout << in[i] << " ";
  }
  std::cout << "\nres=" << res << std::endl;
  */

  assert_almost_eq(res, T(2.25));
}

template <class T>
void test_dot_product_other() {
  constexpr size_t SIZE = 4;
  std::array<T, SIZE> in1{1, 2, 3, 4};
  std::array<T, SIZE> in2{2, 2, 1, 0.5};
  T res;

  {
    cl::sycl::queue& q = create_queue();
    ml::vector_t<T> sycl_vec1(in1.data(), cl::sycl::range<1>(in1.size()));
    sycl_vec1.set_final_data(nullptr);
    ml::vector_t<T> sycl_vec2(in2.data(), cl::sycl::range<1>(in2.size()));
    sycl_vec2.set_final_data(nullptr);
    res = ml::sycl_dot_product(q, sycl_vec1, sycl_vec2);
    clear_eigen_device();
  }

  /*
  for (unsigned i = 0; i < SIZE; ++i) {
    std::cout << in1[i] << " ";
  }
  std::cout << std::endl;
  for (unsigned i = 0; i < SIZE; ++i) {
    std::cout << in2[i] << " ";
  }
  std::cout << "\nres=" << res << std::endl;
  */

  assert_almost_eq(res, T(11));
}

template <class T>
void test_all() {
  test_dot_product_self<T>();
  test_dot_product_other<T>();
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
