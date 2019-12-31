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
#ifndef INCLUDE_ML_MATH_SVD_HPP
#define INCLUDE_ML_MATH_SVD_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "ml/math/mat_mul.hpp"
#include "ml/math/vec_ops.hpp"

namespace ml {

namespace detail {

template <class T>
void stabilize_vec(queue& q, vector_t<T>& prev_vec, vector_t<T>& act_vec,
                   T epsilon) {
  T a = sycl_dot_product(q, prev_vec, act_vec);

  if (a > epsilon) {
    vec_binary_op(q, act_vec, prev_vec, functors::amortize<T>(a));
  }
}

class ml_svd_deflate;
template <class T>
event deflate(queue& q, matrix_t<T>& data, vector_t<T>& act_U_col,
              vector_t<T>& act_V_row, T act_eig_val) {
  return q.submit([&data, &act_U_col, &act_V_row, act_eig_val](handler& cgh) {
    auto v_acc = act_U_col.template get_access_1d<access::mode::read>(cgh);
    auto u_acc = act_V_row.template get_access_1d<access::mode::read>(cgh);
    auto data_acc = data.template get_access_2d<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_svd_deflate, T>>(
        data.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          data_acc(row, col) -= act_eig_val * v_acc(row) * u_acc(col);
        });
  });
}

template <bool Enable>
struct write_vec {
  template <data_dim D, class T>
  static inline void apply(queue&, matrix_t<T>&, vector_t<T>&,
                           const nd_range<1>&, SYCLIndexT) {}
};

template <>
template <data_dim D, class T>
inline void write_vec<true>::apply(queue& q, matrix_t<T>& matrix,
                                   vector_t<T>& vec, const nd_range<1>& nd_rng,
                                   SYCLIndexT row_col) {
  copy_vec_to_mat<D, access::mode::discard_write>(q, matrix, vec, nd_rng,
                                                  row_col);
}

template <bool Enable>
struct write_l {
  template <class T>
  static inline void apply(std::vector<T>&, T, SYCLIndexT) {}
};

template <>
template <class T>
inline void write_l<true>::apply(std::vector<T>& L, T host_l, SYCLIndexT k) {
  L[k] = host_l;
}

}  // namespace detail

/**
 * @brief Output of the SVD.
 *
 * @see svd
 * @tparam T
 */
template <class T>
struct svd_out {
  matrix_t<T> U;
  std::vector<T> L;
  matrix_t<T> V;
  T eig_vals_sum;
};

/**
 * @brief SVD decomposition of the given matrix of size nxp.
 *
 * svd(A) computes the matrices U, S and V such that A = U.S.V.
 * In svd_out, L is the diagonal of S.
 *
 * This implementation is a thin svd meaning that V is of size nxp instead of
 * nxn (the remaining columns being 0). p is assumed to be a power of 2.
 *
 * \p data is modified and it becomes the residual matrix (should have values
 * close to 0). The data should have values in a range bigger than [0, 1] to
 * avoid division by zero.
 *
 * @tparam WriteU whether to write U in the output
 * @tparam WriteL whether to write L in the output
 * @tparam WriteV whether to write V in the output
 * @tparam T
 * @param q
 * @param[in, out] data
 * @param nb_vec if non-zero truncate the svd to compute less eigenpairs
 * @param epsilon threshold below which the eigenvalue is set to zero and
 * ignored
 * @param max_nb_iter if reached, stop the iteration as soon as the difference
 * with the last eigenvalue increases.
 * @return a structure with all the required buffers filled
 */
template <bool WriteU, bool WriteL, bool WriteV, class T>
svd_out<T> svd(queue& q, matrix_t<T>& data, SYCLIndexT nb_vec = 0,
               T epsilon = 1E-4, SYCLIndexT max_nb_iter = 100) {
  auto nb_obs = data.data_range[0];
  auto data_dim = data.data_range[1];
  auto nb_obs_pow2 = to_pow2(nb_obs);
  auto data_dim_pow2 = to_pow2(data_dim);

  if (nb_vec == 0) {
    nb_vec = data_dim;
  }

  nd_range<1> nd_data_dim_pow2_range = get_optimal_nd_range(data_dim_pow2);
  nd_range<1> nd_nb_obs_range = get_optimal_nd_range(nb_obs);
  nd_range<1> nd_nb_obs_pow2_range = get_optimal_nd_range(nb_obs_pow2);
  nd_range<1> nd_nb_vec_range = get_optimal_nd_range(nb_vec);

  svd_out<T> out{matrix_t<T>(range<2>(nb_obs, nb_vec)), std::vector<T>(),
                 matrix_t<T>(range<2>(nb_vec, data_dim),
                             get_optimal_nd_range(nb_vec, data_dim_pow2)),
                 0.0f};
  auto& U = out.U;
  auto& L = out.L;  // diag of S
  auto& V = out.V;
  T& eig_vals_sum = out.eig_vals_sum;

  vector_t<T> prev_U_col(range<1>(nb_obs), nd_nb_obs_pow2_range);
  vector_t<T> act_U_col(range<1>(nb_obs), nd_nb_obs_pow2_range);
  vector_t<T> prev_V_row(range<1>(data_dim), nd_data_dim_pow2_range);
  vector_t<T> act_V_row(range<1>(data_dim), nd_data_dim_pow2_range);

  sycl_memset(q, U);
  sycl_memset(q, V);
  sycl_memset(q, act_U_col);
  sycl_memset(q, act_V_row);

  if (WriteL) {
    L.resize(nb_vec);
  }

  T norm_v;
  T prev_l;
  T act_l{0};
  T prev_diff{0};
  T act_diff;
  SYCLIndexT act_nb_iter;

  for (SYCLIndexT k = 0; k < nb_vec; ++k) {
    copy_mat_to_vec<COL, access::mode::discard_write>(q, data, act_U_col,
                                                      nd_nb_obs_range, k);

    prev_l = 0;
    act_nb_iter = 0;
    while (true) {
      mat_mul<TR>(q, data, act_U_col, act_V_row);

      if (k > 0) {
        detail::stabilize_vec(q, prev_V_row, act_V_row, epsilon);
      }
      norm_v = sycl_norm(q, act_V_row);

      if (norm_v < epsilon) {
        sycl_memset(q, act_V_row);
        sycl_memset(q, act_U_col);
        act_l = 0.0;
      } else {
        sycl_normalize(q, act_V_row, norm_v);
        mat_mul(q, data, act_V_row, act_U_col);

        if (k > 0) {
          detail::stabilize_vec(q, prev_U_col, act_U_col, epsilon);
        }

        act_l = sycl_norm(q, act_U_col);
        assert_real(act_l);
      }

      act_diff = std::fabs(prev_l - act_l);

      // Very verbose log
      /*
      std::cout << "#" << k;
      std::cout << "\t act_nb_iter=" << act_nb_iter;
      std::cout << "\t act_diff=" << act_diff;
      std::cout << "\t eigenvalue=" << act_l << " / " << eig_vals_sum;
      std::cout << std::endl;
      */

      detail::write_l<WriteL>::apply(L, act_l, k);
      if (act_l < epsilon) {
        sycl_memset(q, act_U_col);
      } else {
        sycl_normalize(q, act_U_col, act_l);
      }

      if (act_diff < epsilon ||
          (act_nb_iter > max_nb_iter && act_diff > prev_diff)) {
        break;
      }

      prev_l = act_l;
      prev_diff = act_diff;
      ++act_nb_iter;
    }

    // Verbose log
    /*
    std::cout << "#" << k << "\t act_nb_iter=" << act_nb_iter;
    if (nb_obs_pow2 == data_dim_pow2) { // If the input is symmetric
      // u_v_dist should be close to 0
      std::cout << "\t u_v_dist=" << sycl_dist_no_direction(q,
    act_U_col,act_V_row);
    }
    std::cout << "\t eigenvalue=" << act_l << " / " << eig_vals_sum
              << std::endl;
    */

    if (act_l >= epsilon) {
      eig_vals_sum += act_l;
      detail::deflate(q, data, act_U_col, act_V_row, act_l);
    }

    if (k < nb_vec - 1) {
      sycl_copy(q, act_U_col, prev_U_col);
      sycl_copy(q, act_V_row, prev_V_row);
    }
    detail::write_vec<WriteU>::template apply<COL>(q, U, act_U_col,
                                                   nd_nb_obs_range, k);
    // L has as many rows as V does but is a vector so use its
    // optimal_kernel_range(nb_vec) instead of recomputing it.
    detail::write_vec<WriteV>::template apply<ROW>(q, V, act_V_row,
                                                   nd_nb_vec_range, k);
  }

  return out;
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_SVD_HPP
