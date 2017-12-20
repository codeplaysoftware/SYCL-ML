#ifndef INCLUDE_ML_MATH_TRI_SOLVE_HPP
#define INCLUDE_ML_MATH_TRI_SOLVE_HPP

#include "ml/math/mat_ops.hpp"

namespace ml
{

template <data_dim, data_dim>
class ml_mat_tri_solve;
class ml_mat_tri_solve_div_row;

namespace detail
{

template <data_dim D>
struct tri_solve_data_dim;

// Upper specific case
template <>
struct tri_solve_data_dim<LIN> {
  static inline SYCLIndexT get_row_idx(SYCLIndexT n, SYCLIndexT i) { return n - i - 1; }
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
void div_row(queue& q, matrix_t<T>& A, matrix_t<T>& X, SYCLIndexT row_idx, const nd_range<1>& col_ker_rng) {
  q.submit([&](handler& cgh) {
    auto a_acc = A.template get_access_2d<access::mode::read>(cgh); // Don't need DA because we only access the diagonal
    auto x_acc = X.template get_access_2d<access::mode::read_write, DX>(cgh);
    cgh.parallel_for<NameGen<DX, ml_mat_tri_solve_div_row, T>>(col_ker_rng, [=](nd_item<1> item) {
      auto col = item.get_global(0);
      x_acc(row_idx, col) /= a_acc(row_idx, row_idx);
    });
  });
}

template <data_dim DA, data_dim DX, class T>
void compute_x(queue& q, matrix_t<T>& A, matrix_t<T>& X, SYCLIndexT row_idx) {
  const auto apply_subtract_condition = typename detail::tri_solve_data_dim<DA>::apply_subtract_condition_op();
  q.submit([&](handler& cgh) {
    auto a_acc = A.template get_access_2d<access::mode::read, DA>(cgh);
    auto x_acc = X.template get_access_2d<access::mode::read_write, DX>(cgh);
    cgh.parallel_for<NameGen<0, ml_mat_tri_solve<DA, DX>, T>>(X.get_nd_range(), [=](nd_item<2> item) {
      auto row = item.get_global(DX);
      auto col = item.get_global(opp<DX>());
      if (apply_subtract_condition(row, row_idx))
        x_acc(row, col) -= x_acc(row_idx, col) * a_acc(row, row_idx);
    });
  });
}

} // detail

/**
 * @brief Compute X = A \ B = inv(A) * B without explicitly inverting A.
 *
 * Assumes that A is upper triangular.
 * B (resp. B') must have the same number of rows than A (according to DX).
 * Here we assume that matrix X has been initialized by B beforehand.
 *
 * @tparam DX whether to transpose \p X
 * @tparam DA whether to transpose \p A
 * @tparam T
 * @param q
 * @param[in, out] X
 * @param[in] A
 */
template <data_dim DX = LIN, data_dim DA = LIN, class T>
void tri_solve(queue& q, matrix_t<T>& X, matrix_t<T>& A) {
  const auto n = access_ker_dim(A, 0);
  A.assert_square();
  assert_eq(access_ker_dim<DX>(X, 0), n);

  const auto nb_cols = access_ker_dim<DX>(X, 1);
  const auto col_ker_rng = get_optimal_nd_range(nb_cols);
  const auto get_next_row_idx = typename detail::tri_solve_data_dim<DA>::get_next_row_idx_op();

  // First iteration can be computed directly
  SYCLIndexT row_idx = detail::tri_solve_data_dim<DA>::get_row_idx(n, 0);
  SYCLIndexT next_row_idx = get_next_row_idx(row_idx, 1);
  detail::div_row<DX>(q, A, X, row_idx, col_ker_rng);

  // Each result found must be subtracted for the next iterations
  for (SYCLIndexT i = 1; i < n; ++i) {
    detail::compute_x<DA, DX>(q, A, X, row_idx);
    row_idx = next_row_idx;
    next_row_idx = get_next_row_idx(row_idx, 1);
    detail::div_row<DX>(q, A, X, row_idx, col_ker_rng);
  }
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
 */
template <data_dim DX = LIN, data_dim DA = LIN, class T>
inline void tri_solve(queue& q, matrix_t<T>& X, matrix_t<T>& A, matrix_t<T>& B) {
  sycl_copy(q, B, X);
  tri_solve<DX, DA>(q, X, A);
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
 */
template <data_dim DX = LIN, class T>
void chol_solve(queue& q, matrix_t<T>& X, matrix_t<T>& A, matrix_t<T>& B) {
  tri_solve<DX, TR>(q, X, A, B);
  tri_solve<DX, LIN>(q, X, A);
}

}

#endif //INCLUDE_ML_MATH_TRI_SOLVE_HPP
