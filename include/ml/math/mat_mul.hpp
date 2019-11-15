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
#ifndef INCLUDE_ML_MATH_MAT_MUL_HPP
#define INCLUDE_ML_MATH_MAT_MUL_HPP

#include "ml/math/vec_ops.hpp"

namespace ml
{

/**
 * @brief Matrix multiplication using Eigen.
 *
 * The tensors are sliced to their data_range first in case their kernel_range is bigger.
 *
 * @tparam D1 whether to transpose \p b1
 * @tparam D2 whether to transpose \p b2
 * @tparam T
 * @tparam DIM1 Tensor dimension of \p b1
 * @tparam DIM2 Tensor dimension of \p b2
 * @tparam DIM3 Tensor dimension of \p b3
 * @param[in] b1 mxk
 * @param[in] b2 kxn
 * @param[out] b3 mxn
 */
template <data_dim D1 = LIN, data_dim D2 = LIN, class T, int DIM1, int DIM2, int DIM3>
void mat_mul(queue&, buffer_t<T, DIM1>& b1, buffer_t<T, DIM2>& b2, buffer_t<T, DIM3>& b3) {
  STATIC_ASSERT_DATA_DIM_FOR_DIM_2(DIM1, D1);
  STATIC_ASSERT_DATA_DIM_FOR_DIM_2(DIM2, D2);
  static_assert(1 <= DIM1 && DIM1 <= 2, "");
  static_assert(1 <= DIM1 && DIM2 <= 2, "");
  static_assert(DIM3 == std::min(DIM1, DIM2), "");

  // Act as if data_dim were LIN because the transpose is handled by dims
  // Force dim 2 because SYCL Eigen does not have all possible contractions
  auto eig_t1 = sycl_to_eigen<DIM1, 2>(b1);
  auto eig_t2 = sycl_to_eigen<DIM2, 2>(b2);
  auto eig_t3 = sycl_to_eigen<DIM3, 2>(b3);

  auto sliced_t1 = eig_t1.tensor().slice(eig_dsize_t<2>{0, 0}, detail::range_to_dsize<DIM1, 2>(b1.data_range));
  auto sliced_t2 = eig_t2.tensor().slice(eig_dsize_t<2>{0, 0}, detail::range_to_dsize<DIM2, 2>(b2.data_range));
  auto sliced_t3 = eig_t3.tensor().slice(eig_dsize_t<2>{0, 0}, detail::range_to_dsize<DIM3, 2>(b3.data_range));

  sliced_t3.device(get_eigen_device()) = sliced_t1.contract(sliced_t2, get_contract_dim<opp<D1>(), D2>());
}

template <data_dim, data_dim>
class ml_simple_mat_mul;

/**
 * @brief Naive matrix multiplication.
 *
 * Good enough if k is small.
 *
 * @tparam D1 whether to transpose \p b1
 * @tparam D2 whether to transpose \p b2
 * @tparam T
 * @param q
 * @param[in] m1
 * @param[in] m2
 * @param[out] m3
 */
template <data_dim D1 = LIN, data_dim D2 = LIN, class T>
void simple_mat_mul(queue& q, matrix_t<T>& m1, matrix_t<T>& m2, matrix_t<T>& m3) {
  auto m = m3.get_kernel_range()[0];
  auto n = m3.get_kernel_range()[1];
  auto k = access_data_dim<D1>(m1, 1);

  m3.assert_data_eq_ker();
  assert_eq(access_data_dim<D2>(m2, 0), k);
  assert_rng_less_or_eq<D1>(m1.data_range, m, k);
  assert_rng_less_or_eq<D2>(m2.data_range, k, n);
  assert_rng_less_or_eq(m3.get_kernel_range(), m, n);

  q.submit([&](handler& cgh) {
    auto m1_acc = m1.template get_access_2d<access::mode::read, D1>(cgh);
    auto m2_acc = m2.template get_access_2d<access::mode::read, D2>(cgh);
    auto m3_acc = m3.template get_access_2d<access::mode::discard_write>(cgh);

    cgh.parallel_for<NameGen<0, ml_simple_mat_mul<D1, D2>, T>>(m3.get_nd_range(), [=](nd_item<2> item) {
      auto row = item.get_global_id(0);
      auto col = item.get_global_id(1);
      T sum{0};
      for (SYCLIndexT i = 0; i < k; ++i) {
        sum += m1_acc(row, i) * m2_acc(i, col);
      }
      m3_acc(row, col) = sum;
    });
  });
}

class ml_simple_mat_mul_vec;

/**
 * @brief Naive matrix-vector multiplication.
 *
 * Good enough if k is small.
 *
 * @tparam D whether to transpose \p matrix
 * @tparam T
 * @param q
 * @param[in] matrix mxk
 * @param[in] vec kx1
 * @param[out] out mx1
 */
template <data_dim D = LIN, class T>
void simple_mat_mul_vec(queue& q, matrix_t<T>& matrix, vector_t<T>& vec, vector_t<T>& out) {
  auto m = out.get_kernel_range()[0];
  auto k = vec.data_range[0];

  assert_eq(access_data_dim<D>(matrix, 0), m);
  assert_eq(access_data_dim<D>(matrix, 1), k);

  q.submit([&](handler& cgh) {
    auto mat_acc = matrix.template get_access_2d<access::mode::read, D>(cgh);
    auto vec_acc = vec.template get_access_1d<access::mode::read>(cgh);
    auto out_acc = out.template get_access_1d<access::mode::discard_write>(cgh);

    cgh.parallel_for<NameGen<D, ml_simple_mat_mul_vec, T>>(out.get_nd_range(), [=](nd_item<1> item) {
      auto row = item.get_global_id(0);
      T sum{0};
      for (size_t i = 0; i < k; ++i) {
        sum += mat_acc(row, i) * vec_acc(i);
      }
      out_acc(row) = sum;
    });
  });
}

} // ml

#endif //INCLUDE_ML_MATH_MAT_MUL_HPP
