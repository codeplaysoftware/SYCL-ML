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
#ifndef INCLUDE_ML_MATH_TRI_SOLVE_HPP
#define INCLUDE_ML_MATH_TRI_SOLVE_HPP

#include "ml/math/mat_ops.hpp"

namespace ml {

class ml_mat_tri_solve;
class ml_mat_tri_solve_div_row;

namespace detail {

template <data_dim D>
struct tri_solve_data_dim;

// Upper specific case
template <>
struct tri_solve_data_dim<LIN> {
  static inline SYCLIndexT get_row_idx(SYCLIndexT n, SYCLIndexT i) {
    return n - i - 1;
  }
  using get_next_row_idx_op = std::minus<SYCLIndexT>;
  using apply_subtract_condition_op = std::less<SYCLIndexT>;
};

// Lower specific case
template <>
struct tri_solve_data_dim<TR> {
  static inline SYCLIndexT get_row_idx(SYCLIndexT, SYCLIndexT i) { return i; }
  using get_next_row_idx_op = std::plus<SYCLIndexT>;
  using apply_subtract_condition_op = std::greater<SYCLIndexT>;
};

template <data_dim DX, class T>
event div_row(queue& q, matrix_t<T>& A, matrix_t<T>& X, SYCLIndexT row_idx,
              const nd_range<1>& col_ker_rng) {
  return q.submit([&A, &X, row_idx, col_ker_rng](handler& cgh) {
    // Don't need DA because we only access the diagonal
    auto a_acc = A.template get_access_2d<access::mode::read>(cgh);
    auto x_acc = X.template get_access_2d<access::mode::read_write, DX>(cgh);
    cgh.parallel_for<NameGen<DX, ml_mat_tri_solve_div_row, T>>(
        col_ker_rng, [=](nd_item<1> item) {
          auto col = item.get_global_id(0);
          x_acc(row_idx, col) /= a_acc(row_idx, row_idx);
        });
  });
}

template <data_dim DA, data_dim DX, class T>
event compute_x(queue& q, matrix_t<T>& A, matrix_t<T>& X, SYCLIndexT row_idx) {
  return q.submit([&A, &X, row_idx](handler& cgh) {
    auto a_acc = A.template get_access_2d<access::mode::read, DA>(cgh);
    auto x_acc = X.template get_access_2d<access::mode::read_write, DX>(cgh);
    const auto apply_subtract_condition =
        typename detail::tri_solve_data_dim<DA>::apply_subtract_condition_op();
    cgh.parallel_for<NameGen<DA * 2 + DX, ml_mat_tri_solve, T>>(
        X.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(DX);
          auto col = item.get_global_id(opp<DX>());
          if (apply_subtract_condition(row, row_idx)) {
            x_acc(row, col) -= x_acc(row_idx, col) * a_acc(row, row_idx);
          }
        });
  });
}

}  // namespace detail

/**
 * @brief Compute X = A \ X = inv(A) * X without explicitly inverting A.
 *
 * Assumes that A is upper triangular.
 * X (resp. X') must have the same number of rows than A if DX=LIN (resp.
 * DX=COL)
 *
 * @tparam DX whether to transpose \p X
 * @tparam DA whether to transpose \p A
 * @tparam T
 * @param q
 * @param[in, out] X
 * @param[in] A
 * @return A SYCL event corresponding to the last submitted operation
 */
template <data_dim DX = LIN, data_dim DA = LIN, class T>
event tri_solve(queue& q, matrix_t<T>& X, matrix_t<T>& A) {
  const auto n = access_ker_dim(A, 0);
  A.assert_square();
  assert_eq(access_ker_dim<DX>(X, 0), n);

  const auto nb_cols = access_ker_dim<DX>(X, 1);
  const auto col_ker_rng = get_optimal_nd_range(nb_cols);
  const auto get_next_row_idx =
      typename detail::tri_solve_data_dim<DA>::get_next_row_idx_op();

  // First iteration can be computed directly
  event event;
  SYCLIndexT row_idx = detail::tri_solve_data_dim<DA>::get_row_idx(n, 0);
  SYCLIndexT next_row_idx = get_next_row_idx(row_idx, 1);
  event = detail::div_row<DX>(q, A, X, row_idx, col_ker_rng);

  // Each result found must be subtracted for the next iterations
  for (SYCLIndexT i = 1; i < n; ++i) {
    detail::compute_x<DA, DX>(q, A, X, row_idx);
    row_idx = next_row_idx;
    next_row_idx = get_next_row_idx(row_idx, 1);
    event = detail::div_row<DX>(q, A, X, row_idx, col_ker_rng);
  }
  return event;
}

/**
 * @brief Compute X = A \ B = inv(A) * B without explicitly inverting A.
 *
 * @see tri_solve(queue&, matrix_t<T>&, matrix_t<T>&)
 * @tparam DX whether to transpose \p X
 * @tparam DA whether to transpose \p A
 * @tparam T
 * @param q
 * @param[out] X
 * @param[in] A
 * @param[in] B
 * @return A SYCL event corresponding to the last submitted operation
 */
template <data_dim DX = LIN, data_dim DA = LIN, class T>
inline event tri_solve(queue& q, matrix_t<T>& X, matrix_t<T>& A,
                       matrix_t<T>& B) {
  sycl_copy(q, B, X);
  return tri_solve<DX, DA>(q, X, A);
}

/**
 * @brief Compute X = C \ B = inv(C) * B with C = A'*A.
 *
 * @tparam DX whether to transpose \p X
 * @tparam T
 * @param q
 * @param[out] X
 * @param[in] A
 * @param[in] B
 * @return A SYCL event corresponding to the last submitted operation
 */
template <data_dim DX = LIN, class T>
event chol_solve(queue& q, matrix_t<T>& X, matrix_t<T>& A, matrix_t<T>& B) {
  tri_solve<DX, TR>(q, X, A, B);
  return tri_solve<DX, LIN>(q, X, A);
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_TRI_SOLVE_HPP
