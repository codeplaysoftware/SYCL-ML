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

#include "ml/classifiers/svm/svm.hpp"
#include "utils/utils.hpp"

template <class T>
void test_argmin_cond() {
  constexpr auto NB_ELT = 128LU;
  constexpr auto EXPECTED_MIN_IDX = 13LU;
  constexpr auto TRUE_MIN_IDX = 15LU;
  std::array<T, NB_ELT> host_data;
  fill_random(host_data, 0, 100);
  host_data[EXPECTED_MIN_IDX] = -1;
  host_data[TRUE_MIN_IDX] = -2;

  unsigned long min_idx;
  {
    cl::sycl::queue& q = create_queue();
    ml::vector_t<T> sycl_data(host_data.data(), cl::sycl::range<1>(NB_ELT));
    ml::vector_t<T> sycl_cond((cl::sycl::range<1>(NB_ELT)));

    ml::sycl_memset(q, sycl_cond, T(true));
    // Ignore this index so it should not be returned
    sycl_cond.write_from_host(TRUE_MIN_IDX, false);

    {
      ml::vector_t<ml::SYCLIndexT> device_scalar(ml::range<1>(1));
      auto eig_scalar = ml::sycl_to_eigen<1, 0>(device_scalar);
      bool found =
          ml::detail::argmin_cond(q, sycl_cond, sycl_data, eig_scalar, min_idx);
      assert(found);
    }

    sycl_data.set_final_data(nullptr);
    clear_eigen_device();
  }

  assert_eq(min_idx, EXPECTED_MIN_IDX);
}

int main() {
  try {
    test_argmin_cond<float>();
#ifdef SYCLML_TEST_DOUBLE
    test_argmin_cond<double>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
