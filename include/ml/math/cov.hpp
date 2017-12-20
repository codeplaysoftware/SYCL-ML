#ifndef INCLUDE_ML_MATH_COV_HPP
#define INCLUDE_ML_MATH_COV_HPP

#include "ml/math/mat_mul.hpp"

namespace ml
{

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

} // ml

#endif //INCLUDE_ML_MATH_COV_HPP
