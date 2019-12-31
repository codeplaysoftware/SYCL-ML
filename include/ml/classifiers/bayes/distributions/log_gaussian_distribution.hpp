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
#ifndef INCLUDE_ML_CLASSIFIERS_BAYES_DISTRIBUTIONS_LOG_GAUSSIAN_DISTRIBUTION_HPP
#define INCLUDE_ML_CLASSIFIERS_BAYES_DISTRIBUTIONS_LOG_GAUSSIAN_DISTRIBUTION_HPP

#include "ml/math/helper.hpp"
#include "ml/math/qr.hpp"
#include "ml/math/tri_solve.hpp"

namespace ml {

class ml_gd_normalize_r;
class ml_gd_normalize_random_r;
class ml_gd_get_plx_data;

/**
 * @brief Log of a Gaussian Distribution
 *
 * Used both to compute the parameters of such a distribution (i.e. the average
 * and the covariance matrix) and to compute a probability given these
 * parameters.
 *
 * Instead of computing the covariance matrix C, the R of the qr decomposition
 * is computed. This allow to compute the inverse of C faster (because R is
 * triangular) and with less errors. This is possible because for a dataset A:\n
 * \f$
 * A = Q * R \\
 * C = (A' * A) / N \\
 * C = ((R' * Q') * (Q * R)) / N \\
 * C = (R' * R) / N \\
 * \f$
 * \n Because Q is orthogonal
 *
 * Then there is no need to compute the upper Cholesky decomposition U of C to
 * invert C because U = R / sqrt(N) assuming that all values on the diagonal of
 * R are positive.
 *
 * @tparam T
 */
template <class T>
class log_gaussian_distribution {
 public:
  using DataType = T;
  static const T LOG_2_PI;

  /**
   * @brief Compute the parameters of a Gaussian Distribution
   *
   * @param q
   * @param[in] act_data is modified during this call and shouldn't be used
   * afterward
   * @param[out] data_avg
   * @param[out] act_r R such that C=R'.R
   * @param[out] log_cov_det log of the determinant of C
   * @param[in] weight by default is the number of observation given
   * @param[in] plx_k optional vector used to give a weight to each observation
   */
  void compute(queue& q, matrix_t<T>& act_data, vector_t<T>& data_avg,
               matrix_t<T>& act_r, T& log_cov_det, T weight,
               vector_t<T>& plx_k) {
    bool use_plx = plx_k.get_kernel_size() >= access_ker_dim(act_data, 0);
    matrix_t<T> plx_data(
        use_plx ? act_data.data_range : range<2>(),
        use_plx ? act_data.kernel_range : nd_range<2>(range<2>(), range<2>()));
    if (use_plx) {
      get_plx_data(q, act_data, plx_k, plx_data, functors::identity<T>());
    }
    matrix_t<T>& data = use_plx ? plx_data : act_data;
    avg(q, data, data_avg);

    matrix_t<T> center_data(act_data.data_range, act_data.kernel_range);
    mat_vec_apply_op<COL>(q, act_data, center_data, data_avg, std::minus<T>());

    if (use_plx) {
      get_plx_data(q, center_data, plx_k, plx_data, functors::sqrt<T>());
    }
    data = use_plx ? plx_data : center_data;

    qr(q, data, data_avg.data_range, data_avg.kernel_range);

    T factor = 1 / std::sqrt(weight);
    q.submit([&data, &act_r, factor](handler& cgh) {
      auto old_r_acc = data.template get_access_2d<access::mode::read>(cgh);
      auto new_r_acc =
          act_r.template get_access_2d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGen<0, ml_gd_normalize_r, T>>(
          act_r.get_nd_range(), [=](nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            new_r_acc(row, col) =
                (col >= row) ? old_r_acc(row, col) * factor : 0;
          });
    });

    // get_log_cov_det will block until the result is computed and copied to
    // the host
    log_cov_det = get_log_cov_det(q, act_r);
  }

  /**
   * @brief Compute the log of the probability of the given data with the
   * computed parameters.
   *
   * Uses the multivariate Gaussian formula.
   *
   * @param q
   * @param[in] act_data
   * @param[in] data_avg
   * @param[in] act_r
   * @param log_cov_det
   * @param[out] dist
   * @param data_dim
   */
  void compute_dist(queue& q, matrix_t<T>& act_data, vector_t<T>& data_avg,
                    matrix_t<T>& act_r, T log_cov_det, vector_t<T>& dist,
                    eig_index_t data_dim) {
    assert_less_or_eq(access_ker_dim(act_data, 0), dist.get_kernel_size());

    matrix_t<T> center_data(act_data.data_range, act_data.kernel_range);
    mat_vec_apply_op<COL>(q, act_data, center_data, data_avg, std::minus<T>());

    matrix_t<T> X(act_data.data_range, act_data.kernel_range);
    chol_solve<TR>(q, X, act_r, center_data);
    mat_inplace_binary_op(q, X, center_data, std::multiplies<T>());

    auto eig_contracted = sycl_to_eigen(X);
    auto eig_dist = sycl_to_eigen(dist, range<1>(access_ker_dim(act_data, 0)));
    T cst = data_dim * LOG_2_PI + log_cov_det;
    auto simplified_dist = eig_contracted.tensor().sum(eig_dims_t<1>{1});
    eig_dist.device() = -T(0.5) * (simplified_dist + cst);
  }

  /**
   * @brief Randomize the parameters of a Gaussian Distribution.
   *
   * @param q
   * @param[in] data_sample sample used for \p data_avg
   * @param percent_rnd value in [0, 1] determining how much the \p data_avg is
   * random. The complement is used for the given sample
   * @param[in] offset_noise offset noise for \p data_avg
   * @param[in] range_noise range noise for \p data_avg
   * @param[out] data_avg
   * @param[out] act_r
   * @param[out] log_cov_det
   */
  void randomize(queue& q, vector_t<T>& data_sample, T percent_rnd,
                 vector_t<T>& offset_noise, vector_t<T>& range_noise,
                 vector_t<T>& data_avg, matrix_t<T>& act_r, T& log_cov_det) {
    assert_eq(data_avg.get_kernel_size(), data_sample.get_kernel_size());

    using UniformRandom = Eigen::internal::UniformRandomGenerator<T>;
    {
      auto eig_data_avg = sycl_to_eigen(data_avg);
      auto eig_data_sample = sycl_to_eigen(data_sample);
      auto eig_offset_noise = sycl_to_eigen(offset_noise);
      auto eig_range_noise = sycl_to_eigen(range_noise);

      auto eig_raw_rnd = eig_data_avg.tensor().template random<UniformRandom>();
      auto eig_rnd_mu =
          eig_range_noise.tensor() * eig_raw_rnd + eig_offset_noise.tensor();
      eig_data_avg.device() = eig_data_sample.tensor() * (1 - percent_rnd) +
                              eig_rnd_mu * percent_rnd;
    }

    randomize_r(q, act_r);

    // get_log_cov_det will block until the result is computed and copied to
    // the host
    log_cov_det = get_log_cov_det(q, act_r);
  }

 private:
  /**
   * @brief Get the data modified by op(plx).
   *
   * @tparam Op T -> T
   * @param q
   * @param[in] data
   * @param[in] plx_k
   * @param[out] plx_data
   * @param op
   * @return A SYCL event corresponding to the submitted operation
   */
  template <class Op>
  event get_plx_data(queue& q, matrix_t<T>& data, vector_t<T>& plx_k,
                     matrix_t<T>& plx_data, Op op) {
    return q.submit([&data, &plx_k, &plx_data, op](handler& cgh) {
      auto data_acc = data.template get_access_2d<access::mode::read>(cgh);
      auto plx_acc = plx_k.template get_access_1d<access::mode::read>(cgh);
      auto plx_data_acc =
          plx_data.template get_access_2d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGen<0, ml_gd_get_plx_data, T, Op>>(
          plx_data.get_nd_range(), [=](nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            plx_data_acc(row, col) = data_acc(row, col) * op(plx_acc(row));
          });
    });
  }

  /**
   * @brief Compute the log of the determinant of C from the R matrix.
   *
   * \f$ log(det(C)) = log(det(R)^2) = log(prod(diag(R))^2) = 2 *
   * sum(log(abs(diag(R)))) \f$
   *
   * @param q
   * @param act_r
   * @return \f$ log(det(C)) \f$
   */
  T get_log_cov_det(queue& q, matrix_t<T>& act_r) {
    T log_cov_det = 2 * reduce_diag<functors::sum_log_abs<T>>(q, act_r);
    assert_real(log_cov_det);
    return log_cov_det;
  }

  /**
   * @brief Randomize the R matrix.
   *
   * The constants used here are to make the randomized matrix looks like a real
   * R matrix. It also makes sure the determinant is not too close to 0.
   *
   * @param q
   * @param[out] act_r
   * @return A SYCL event corresponding to the submitted operation
   */
  event randomize_r(queue& q, matrix_t<T>& act_r) {
    using UniformRandom = Eigen::internal::UniformRandomGenerator<T>;
    {
      auto eig_r = sycl_to_eigen(act_r);
      eig_r.device() = eig_r.tensor().template random<UniformRandom>();
    }

    return q.submit([&act_r](handler& cgh) {
      auto act_r_acc =
          act_r.template get_access_2d<access::mode::read_write>(cgh);
      cgh.parallel_for<NameGen<0, ml_gd_normalize_random_r, T>>(
          act_r.get_nd_range(), [=](nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            auto& act_rc = act_r_acc(row, col);
            if (row > col) {
              act_rc = 0;
            } else if (row < col) {
              act_rc = (act_rc * T(3) - T(1.5)) /
                       (cl::sycl::sqrt(static_cast<T>(row * col)) + T(1));
            } else {
              act_rc = (((act_rc - T(0.5)) >= T(0)) * T(2) - T(1)) *
                       (act_rc + T(0.5) + T(1) / (row + 1));
            }
          });
    });
  }
};

template <class T>
const T log_gaussian_distribution<T>::LOG_2_PI = std::log(2 * ml::PI<T>);

/**
 * @brief Wrapper around log_gaussian_distribution that stores some parameters.
 *
 * @see log_gaussian_distribution
 * @tparam T
 */
template <class T>
class buffered_log_gaussian_distribution : public log_gaussian_distribution<T> {
 public:
  buffered_log_gaussian_distribution()
      : _data_dim(),
        _nb_obs_rng(),
        _nb_obs_pow2_rng(_nb_obs_rng, _nb_obs_rng),
        _data_avg(),
        _act_r(),
        _log_cov_det() {}

  void init(const range<1>& data_dim_rng, const nd_range<1>& data_dim_pow2_rng,
            const range<2>& data_dim_rng_d2,
            const nd_range<2>& data_dim_pow2_rng_d2) {
    _data_dim = data_dim_rng[0];

    _fake_plx_k = vector_t<T>(range<1>());
    _data_avg = vector_t<T>(data_dim_rng, data_dim_pow2_rng);
    _act_r = matrix_t<T>(data_dim_rng_d2, data_dim_pow2_rng_d2);
  }

  inline void compute(queue& q, matrix_t<T>& act_data) {
    compute(q, act_data, access_data_dim(act_data, 0), _fake_plx_k);
  }

  inline void compute(queue& q, matrix_t<T>& act_data, T weight,
                      vector_t<T>& plx_k) {
    log_gaussian_distribution<T>::compute(q, act_data, _data_avg, _act_r,
                                          _log_cov_det, weight, plx_k);
  }

  inline void compute_dist(queue& q, matrix_t<T>& act_data, vector_t<T>& dist) {
    log_gaussian_distribution<T>::compute_dist(q, act_data, _data_avg, _act_r,
                                               _log_cov_det, dist, _data_dim);
  }

  inline void randomize(queue& q, vector_t<T>& data_sample, T percent_rnd,
                        vector_t<T>& offset_noise, vector_t<T>& range_noise) {
    log_gaussian_distribution<T>::randomize(q, data_sample, percent_rnd,
                                            offset_noise, range_noise,
                                            _data_avg, _act_r, _log_cov_det);
  }

  inline void randomize(queue& q, buffered_log_gaussian_distribution<T>& other,
                        T percent_rnd, vector_t<T>& offset_noise,
                        vector_t<T>& range_noise) {
    randomize(q, other._data_avg, percent_rnd, offset_noise, range_noise);
  }

  void load_from_disk(queue& q, const std::string& prefix = "") {
    load_array(q, _data_avg, prefix + "_data_avg");
    load_array(q, _act_r, prefix + "_act_r");
    load_array(&_log_cov_det, 1, prefix + "_log_cov_det");
  }

  void save_to_disk(queue& q, const std::string& prefix = "") {
    save_array(q, _data_avg, prefix + "_data_avg");
    save_array(q, _act_r, prefix + "_act_r");
    save_array(&_log_cov_det, 1, prefix + "_log_cov_det");
  }

 private:
  eig_index_t _data_dim;
  range<1> _nb_obs_rng;
  nd_range<1> _nb_obs_pow2_rng;

  vector_t<T> _fake_plx_k;
  vector_t<T> _data_avg;
  matrix_t<T> _act_r;
  T _log_cov_det;
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_BAYES_DISTRIBUTIONS_LOG_GAUSSIAN_DISTRIBUTION_HPP
