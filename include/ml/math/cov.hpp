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
#ifndef INCLUDE_ML_MATH_COV_HPP
#define INCLUDE_ML_MATH_COV_HPP

#include "ml/math/mat_mul.hpp"

namespace ml {

/**
 * @brief Compute the covariance matrix of \p dataset
 *
 * Assumes the data has been centered already.
 * It is normalized by the number of observation N (instead of the usual N-1).
 * Formula for D=ROW is \f$ (dataset' * dataset) / N \f$
 *
 * @tparam D specifies which dimension represents the number of observations
 * @tparam T
 * @param q
 * @param[in] dataset
 * @param[out] cov_mat
 */
template <data_dim D = ROW, class T>
void cov(queue& q, matrix_t<T>& dataset, matrix_t<T>& cov_mat) {
  auto nb_obs = access_data_dim<D>(dataset, 0);
  auto data_dim = access_data_dim<D>(dataset, 1);
  assert_rng_eq(cov_mat.data_range, range<2>(data_dim, data_dim));

  mat_mul<opp<D>(), D>(q, dataset, dataset, cov_mat);
  sycl_normalize(q, cov_mat, static_cast<T>(nb_obs));
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_COV_HPP
