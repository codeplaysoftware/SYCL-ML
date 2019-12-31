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
#ifndef INCLUDE_ML_MATH_QR_HPP
#define INCLUDE_ML_MATH_QR_HPP

#include "ml/math/vec_ops.hpp"

namespace ml {

class ml_qr;

/**
 * @brief QR decomposition of the given matrix of size mxn.
 *
 * Uses the Householder transformations algorithm.
 * Note: A blocked Householder would be more performant.
 *
 * qr(A) computes Q and R such that A = Q * R where Q is an orthogonal matrix
 * and R an upper triangular matrix. This implementation assumes that m is
 * greater than n and only writes R in the upper triangular part of A. The lower
 * triangular part of R should be set to 0 if needed. Note that for each row of
 * R a sign can be chosen, this implementation always chooses 1.
 *
 * @tparam T
 * @param q
 * @param[in, out] mat
 * @param w temporary buffer must be of size m at least.
 * @param vec_buf temporary buffer must be of size n at least.
 * @param eps threshold below which the division by u1 is avoided.
 */
template <class T>
void qr(queue& q, matrix_t<T>& mat, vector_t<T>& w, vector_t<T>& vec_buf,
        T eps = 1E-5) {
  auto m = access_ker_dim(mat, 0);
  auto n = access_ker_dim(mat, 1);
  using IndexT = decltype(n);

  assert_less_or_eq(n, m);
  assert_less_or_eq(m, w.data_range[0]);
  assert_less_or_eq(n, vec_buf.data_range[0]);

  static constexpr T ACT_SIGN = 1;
  SYCLIndexT jj_offset;
  T host_mat_jj;
  T act_norm;
  T act_u1;
  T act_tau;

  auto eig_mat = sycl_to_eigen(mat);
  // Force tensor dim 2 for matrix multiplication
  auto eig_w = sycl_to_eigen_2d(w);
  auto eig_vec_buf = sycl_to_eigen_2d(vec_buf);
  vector_t<T> norm_buf((range<1>(1)));
  auto eig_norm = sycl_to_scalar_eigen(norm_buf);
  eig_dsize_t<1> slice_offsets_d1;
  eig_dsize_t<1> slice_extents_d1;
  eig_dsize_t<2> slice_offsets_mat;
  eig_dsize_t<2> slice_extents_mat;
  eig_dsize_t<2> slice_offsets_w{0, 0};
  eig_dsize_t<2> slice_extents_w{1, 1};
  eig_dsize_t<2> slice_offsets_vec_buf{0, 0};
  eig_dsize_t<2> slice_extents_vec_buf{1, 1};

  auto compute_acts = [&](IndexT j) {
    jj_offset = j * (n + 1);
    // Get elements with indices [j, m] of the jth column and take the norm
    slice_offsets_d1[0] = j;
    slice_extents_d1[0] = m - j;
    eig_norm.device() = eig_mat.tensor()
                            .chip(j, 1)
                            .slice(slice_offsets_d1, slice_extents_d1)
                            .square()
                            .sum()
                            .sqrt();
    host_mat_jj = mat.read_to_host(jj_offset);
    // At each iteration the sign can be chosen to be different.
    // Choosing it to be -sign(mat(j,j)) maximizes the value of u1 but is more
    // likely to cause division by zero
    // act_sign = -cl::sycl::sign(host_mat_jj);
    act_norm = ACT_SIGN * norm_buf.read_to_host(0);
    act_u1 = host_mat_jj - act_norm;
    act_tau = -act_u1 / act_norm;
    mat.write_from_host(jj_offset, act_norm);
  };

  auto w_rng = w.kernel_range;
  auto mat_rng = mat.kernel_range;
  SYCLIndexT nb_rows_ker;
  IndexT j = 0;
  for (; j < n - 1; ++j) {
    compute_acts(j);

    if (std::abs(act_u1) < eps) {
      // Note: matrix Q would be inacurate if this is reached
      continue;
    }

    nb_rows_ker = m - j;
    if (nb_rows_ker % 2 == 0) {
      bool nb_rows_ker_is_pow2 = is_pow2(nb_rows_ker);
      if (nb_rows_ker_is_pow2 || !is_pow2(w_rng.get_global_range()[0])) {
        w_rng = get_optimal_nd_range(nb_rows_ker);
      }
      if (nb_rows_ker_is_pow2 || !is_pow2(mat_rng.get_global_range()[0])) {
        mat_rng = get_optimal_nd_range(nb_rows_ker, access_ker_dim(mat, 1));
      }
    }

    // Compute w and update R
    q.submit([&mat, &w, w_rng, nb_rows_ker, act_u1, j](handler& cgh) {
      auto mat_acc = mat.template get_access_2d<access::mode::read_write>(cgh);
      auto w_acc = w.template get_access_1d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGen<0, ml_qr, T>>(w_rng, [=](nd_item<1> item) {
        auto row = item.get_global_id(0) + 1;
        if (row < nb_rows_ker) {
          auto val = mat_acc(row + j, j) / act_u1;
          mat_acc(row + j, j) = val;
          w_acc(row) = val;
        }
      });
    });
    w.write_from_host(0, T(1));

    // Compute vec_buf
    slice_extents_w[0] = nb_rows_ker;
    slice_extents_vec_buf[0] = n - j - 1;
    slice_offsets_mat[0] = j;
    slice_offsets_mat[1] = j + 1;
    slice_extents_mat[0] = nb_rows_ker;
    slice_extents_mat[1] = n - j - 1;
    auto sliced_w = eig_w.tensor().slice(slice_offsets_w, slice_extents_w);
    auto sliced_vec_buf = eig_vec_buf.tensor().slice(slice_offsets_vec_buf,
                                                     slice_extents_vec_buf);
    auto sliced_mat =
        eig_mat.tensor().slice(slice_offsets_mat, slice_extents_mat);
    sliced_vec_buf.device(get_eigen_device()) =
        sliced_mat.contract(sliced_w, get_contract_dim<ROW, ROW>());

    // Update R
    q.submit([&vec_buf, &w, &mat, mat_rng, act_tau, j, m, n](handler& cgh) {
      auto vec_acc = vec_buf.template get_access_1d<access::mode::read>(cgh);
      auto w_acc = w.template get_access_1d<access::mode::read>(cgh);
      auto mat_acc = mat.template get_access_2d<access::mode::read_write>(cgh);
      cgh.parallel_for<NameGen<1, ml_qr, T>>(mat_rng, [=](nd_item<2> item) {
        auto row = item.get_global_id(0);
        auto col = item.get_global_id(1);
        if (row < m - j && col < n - j - 1) {
          mat_acc(j + row, j + 1 + col) -=
              (act_tau * w_acc(row)) * vec_acc(col);
        }
      });
    });
  }

  compute_acts(j);
}

/**
 * @brief QR decomposition of the given matrix.
 *
 * @tparam T
 * @param q
 * @param[in, out] mat
 * @param data_dim_rng 1d range of the size of an observation
 * @param data_dim_pow2_rng 1d kernel range of the size of an observation (can
 * be padded to a bigger power of 2)
 */
template <class T>
void qr(queue& q, matrix_t<T>& mat, const range<1>& data_dim_rng,
        const nd_range<1>& data_dim_pow2_rng) {
  range<1> nb_obs_rng(access_ker_dim(mat, 0));
  auto nb_obs_pow2_rng = get_optimal_nd_range(nb_obs_rng);
  vector_t<T> w_buf(nb_obs_rng, nb_obs_pow2_rng);
  vector_t<T> vec_buf(data_dim_rng, data_dim_pow2_rng);

  qr(q, mat, w_buf, vec_buf);
}

/**
 * @brief QR decomposition of the given matrix.
 *
 * @see qr(queue&, matrix_t<T>&, const range<1>&, const nd_range<1>&, const
 * range<1>&, const nd_range<1>&)
 * @tparam T
 * @param q
 * @param[in, out] mat
 */
template <class T>
void qr(queue& q, matrix_t<T>& mat) {
  range<1> data_dim_rng(access_ker_dim(mat, 1));
  auto data_dim_pow2_rng = get_optimal_nd_range(data_dim_rng);
  qr(q, mat, data_dim_rng, data_dim_pow2_rng);
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_QR_HPP
