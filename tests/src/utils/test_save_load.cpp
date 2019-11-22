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

#include "ml/utils/buffer_t.hpp"
#include "utils/utils.hpp"

template <class T>
void test_save_load_host() {
  constexpr size_t SIZE = 4;
  std::array<T, SIZE> buf{-1, 0, -1.5, 0.5};
  std::array<T, SIZE> res;

  ml::save_array(buf.data(), SIZE, "test_buf");
  ml::load_array(res.data(), SIZE, "test_buf");

  /*
  std::cout << "Saved: ";
  ml::print(buf, 1, SIZE);
  std::cout << "Loaded: ";
  ml::print(res, 1, SIZE);
  */

  assert_vec_almost_eq(res, buf);
}

template <class T>
void test_save_load_device() {
  constexpr size_t SIZE = 6;
  std::array<T, SIZE> buf{-10, 0, -1.5, 3, 3, 1};
  std::array<T, SIZE> res;

  {
    cl::sycl::queue& q = create_queue();
    {
      ml::matrix_t<T> sycl_buf(const_cast<const T*>(buf.data()),
                               cl::sycl::range<2>(2, 3));
      ml::save_array(q, sycl_buf, "test_buf");
    }
    ml::matrix_t<T> sycl_res(cl::sycl::range<2>(2, 3));
    ml::load_array(q, sycl_res, "test_buf");

    sycl_res.set_final_data(res.data());
    clear_eigen_device();
  }

  /*
  std::cout << "Saved: ";
  ml::print(buf, 1, SIZE);
  std::cout << "Loaded: ";
  ml::print(res, 1, SIZE);
  */

  assert_vec_almost_eq(res, buf);
}

template <class T>
void test_all() {
  test_save_load_host<T>();
  test_save_load_device<T>();
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
