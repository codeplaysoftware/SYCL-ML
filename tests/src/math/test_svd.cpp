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

#include "ml/math/mat_mul.hpp"
#include "ml/math/mat_ops.hpp"
#include "ml/math/svd.hpp"
#include "utils/utils.hpp"

template <class T, ml::data_dim D>
void test_svd_general() {
  constexpr auto NB_OBS = 4LU;
  constexpr auto ACT_SIZE_OBS = NB_OBS;
  constexpr auto SIZE_OBS_POW2 = ACT_SIZE_OBS;
  std::array<T, NB_OBS * SIZE_OBS_POW2> host_data{
      1.0, 2.0, 0.0,  -3.0, 2.0,  -5.0, 2.0,  1.0,
      0.0, 2.0, -1.0, -1.0, -3.0, 1.0,  -1.0, 3.0};

  constexpr auto NB_VEC = ACT_SIZE_OBS;

  std::array<T, NB_OBS * NB_VEC> host_V;
  std::array<T, NB_VEC> host_L;
  std::array<T, NB_VEC * SIZE_OBS_POW2> host_U;
  std::array<T, NB_OBS * SIZE_OBS_POW2> host_residual;
  std::array<T, NB_OBS * SIZE_OBS_POW2> host_data_svd;
  std::array<T, NB_OBS * SIZE_OBS_POW2> host_centered_data;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(),
                              cl::sycl::range<2>(NB_OBS, SIZE_OBS_POW2));
    sycl_data.data_range = cl::sycl::range<2>(NB_OBS, ACT_SIZE_OBS);
    ml::vector_t<T> sycl_data_avg(cl::sycl::range<1>(ACT_SIZE_OBS),
                                  ml::get_optimal_nd_range(SIZE_OBS_POW2));

    ml::avg<D>(q, sycl_data, sycl_data_avg);
    ml::center_data<ml::opp<D>()>(q, sycl_data, sycl_data_avg);
    ml::matrix_t<T> sycl_centered_data(sycl_data.data_range,
                                       sycl_data.kernel_range);
    ml::sycl_copy(q, sycl_data, sycl_centered_data);

    auto VLU = ml::svd<true, true, true>(q, sycl_data);
    auto& sycl_U = VLU.U;
    auto& sycl_V = VLU.V;
    auto& vec_L = VLU.L;
    ml::assert_rng_eq({NB_OBS, NB_VEC}, sycl_U.data_range);
    ml::assert_eq(NB_VEC, vec_L.size());
    ml::assert_rng_eq({NB_VEC, ACT_SIZE_OBS}, sycl_V.data_range);

    std::copy(std::begin(vec_L), std::end(vec_L), std::begin(host_L));
    ml::vector_t<T> sycl_L(host_L.data(), cl::sycl::range<1>(host_L.size()));
    sycl_L.set_final_data(nullptr);

    ml::matrix_t<T> sycl_data_svd(sycl_data.data_range, sycl_data.kernel_range);
    ml::matrix_t<T> sycl_copy_V(sycl_V.data_range, sycl_V.kernel_range);
    ml::sycl_copy(q, sycl_V, sycl_copy_V);
    ml::mat_vec_apply_op(q, sycl_copy_V, sycl_L,
                         std::multiplies<T>());  // diag(L) * V
    ml::mat_mul(q, sycl_U, sycl_copy_V, sycl_data_svd);
    ml::mat_inplace_binary_op(q, sycl_data_svd, sycl_data,
                              std::plus<T>());  // Add residual

    sycl_data.set_final_data(host_residual.data());
    sycl_centered_data.set_final_data(host_centered_data.data());
    sycl_data_svd.set_final_data(host_data_svd.data());
    sycl_U.set_final_data(host_U.data());
    sycl_V.set_final_data(host_V.data());
    clear_eigen_device();
  }

  /*
  std::cout << "host data:\n";
  ml::print(host_data, NB_OBS, SIZE_OBS_POW2);
  std::cout << "\nU:\n";
  ml::print(host_U, NB_VEC, SIZE_OBS_POW2);
  std::cout << "\nL:\n";
  ml::print(host_L, 1, NB_VEC);
  std::cout << "\nV:\n";
  ml::print(host_V, NB_OBS, NB_VEC);
  std::cout << "\nR:\n";
  ml::print(host_residual, NB_OBS, SIZE_OBS_POW2);
  std::cout << "\ndata svd:\n";
  ml::print(host_data_svd, NB_OBS, SIZE_OBS_POW2);
  */

  assert_vec_almost_eq(host_centered_data, host_data_svd);
  for (unsigned i = 0; i < NB_OBS * SIZE_OBS_POW2; ++i) {
    assert_almost_eq(host_residual[i], T(0));
  }
}

int main(void) {
  try {
    test_svd_general<float, ml::LIN>();
#ifdef SYCLML_TEST_DOUBLE
    test_svd_general<double, ml::LIN>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
