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

namespace ml {

/**
 * @brief Matrix multiplication using Eigen.
 *
 * The tensors are sliced to their data_range first in case their kernel_range
 * is bigger.
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
template <data_dim D1 = LIN, data_dim D2 = LIN, class T, int DIM1, int DIM2,
          int DIM3>
void mat_mul(queue&, buffer_t<T, DIM1>& b1, buffer_t<T, DIM2>& b2,
             buffer_t<T, DIM3>& b3) {
  STATIC_ASSERT_DATA_DIM_FOR_DIM_2(DIM1, D1);
  STATIC_ASSERT_DATA_DIM_FOR_DIM_2(DIM2, D2);
  static_assert(1 <= DIM1 && DIM1 <= 2, "");
  static_assert(1 <= DIM1 && DIM2 <= 2, "");
  static_assert(DIM3 == std::min(DIM1, DIM2), "");

  // Act as if data_dim were LIN because the transpose is handled by dims
  // Reshape inputs and outputs to be 2D
  auto eig_t1 = sycl_to_eigen<DIM1, 2>(b1);
  auto eig_t2 = sycl_to_eigen<DIM2, 2>(b2);
  auto eig_t3 = sycl_to_eigen<DIM3, 2>(b3);

  auto sliced_t1 = eig_t1.tensor().slice(
      eig_dsize_t<2>{0, 0}, detail::range_to_dsize<DIM1, 2>(b1.data_range));
  auto sliced_t2 = eig_t2.tensor().slice(
      eig_dsize_t<2>{0, 0}, detail::range_to_dsize<DIM2, 2>(b2.data_range));
  auto sliced_t3 = eig_t3.tensor().slice(
      eig_dsize_t<2>{0, 0}, detail::range_to_dsize<DIM3, 2>(b3.data_range));

  sliced_t3.device(get_eigen_device()) =
      sliced_t1.contract(sliced_t2, get_contract_dim<opp<D1>(), D2>());
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_MAT_MUL_HPP
