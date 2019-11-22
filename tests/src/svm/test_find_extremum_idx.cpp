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
void test_find_extremum_idx() {
  constexpr auto NB_ELT = 128LU;
  constexpr auto EXPECTED_MIN_IDX = 13LU;
  std::array<T, NB_ELT> host_data;
  fill_random(host_data, 0, 100);
  host_data[EXPECTED_MIN_IDX] = -1;

  unsigned long min_idx;
  {
    cl::sycl::queue& q = create_queue();
    ml::vector_t<T> sycl_data(host_data.data(), cl::sycl::range<1>(NB_ELT));
    ml::vector_t<T> sycl_cond((cl::sycl::range<1>(NB_ELT)));
    ml::vector_t<uint32_t> start_search_indices(sycl_data.data_range,
                                                sycl_data.kernel_range);
    auto start_search_rng = ml::get_optimal_nd_range(
        start_search_indices.kernel_range.get_global_linear_range() / 2);
    ml::vector_t<uint32_t> buff_search_indices(start_search_rng);

    ml::sycl_memset(q, sycl_cond, T(true));
    ml::sycl_init_func_i(q, start_search_indices,
                         start_search_indices.kernel_range,
                         ml::functors::identity<T>());

    auto find_size_threshold_host = std::min(NB_ELT, 8LU);
    bool found = ml::detail::find_extremum_idx(
        q, sycl_cond, sycl_data, start_search_indices, buff_search_indices,
        start_search_rng, find_size_threshold_host, std::less<T>(), min_idx);
    assert(found);

    sycl_data.set_final_data(nullptr);
    clear_eigen_device();
  }

  assert_eq(min_idx, EXPECTED_MIN_IDX);
}

int main() {
  try {
    test_find_extremum_idx<float>();
#ifdef SYCLML_TEST_DOUBLE
    test_find_extremum_idx<double>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
