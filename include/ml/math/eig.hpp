#ifndef INCLUDE_ML_MATH_EIG_HPP
#define INCLUDE_ML_MATH_EIG_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#include "ml/math/mat_inv.hpp"
#include "ml/math/mat_mul.hpp"
#include "ml/utils/copy.hpp"

namespace ml
{

namespace detail {

/**
 * @brief Rayleigh quotient iteration
 *
 * v(i+1) = (A - Lambda(i)*I)^-1 * v(i) / ||(A - Lambda(i)*I)^-1 * v(i)||
 * Lambda(i) = v(i)' * A * v(i) / v(i)' * v(i)
 *
 *
 * @tparam T
 * @param q
 * @param matrix
 * @param act_eig_vec
 * @param eig_val
 * @param vec_buffer
 * @param mat_buffer
 * @param c_buffer
 * @param block_buffer
 * @param actual_vec_range
 * @param vec_range_pow2
 */
template <class T>
void rayleigh_vec_iteration(queue& q, matrix_t<T>& matrix, vector_t<T>& act_eig_vec, T eig_val,
                            vector_t<T>& vec_buffer, matrix_t<T>& mat_buffer,
                            matrix_t<T>& c_buffer, matrix_t<T>& block_buffer,
                            const nd_range<1>& actual_vec_range, const nd_range<1>& vec_range_pow2) {
  auto data_dim = actual_vec_range.get_global_range(0);
  mat_inv(q, matrix, mat_buffer, c_buffer, block_buffer, data_dim, -eig_val); //TODO inv_solve
  mat_mul_vec(q, mat_buffer, act_eig_vec, vec_buffer, data_dim, data_dim);
  T norm = sycl_norm(q, vec_buffer);

  // Copy normalized vec_buffer in act_eig_vec
  return q.submit([&](handler& cgh) {
    auto tmp_acc = vec_buffer.template get_access<access::mode::read>(cgh);
    auto vec_acc = act_eig_vec.template get_access<access::mode::write>(cgh);
    cgh.parallel_for<NameGen<1, class MLRayleighIter, T>>(vec_range_pow2, [=](nd_item<1> item) {
      auto row = item.get_global(0);
      vec_acc[row] = tmp_acc[row] / norm;
    });
  });
}

// Assumes matrix is symmetric
// A_(i+1) = A_i - Lambda(i) * v(i) * v(i)'
// TODO: try to use the symmetry for optimization
template <class T>
handler_event deflate(queue& q, matrix_t<T>& matrix, vector_t<T>& act_eig_vec, T eig_val,
                      const nd_range<2>& matrix_range) {
  TIME(eig_deflate);
  return q.submit([&](handler& cgh) {
    auto vec_acc = act_eig_vec.template get_access<access::mode::read>(cgh);
    auto matrix_acc = matrix.template get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<0, class MLDeflate, T>>(matrix_range, [=](nd_item<2> item) {
      auto row = item.get_global(0);
      auto col = item.get_global(1);
      matrix_acc[row][col] -= eig_val * vec_acc[row] * vec_acc[col];
    });
  });
}

} // detail

// Compute enough eigenpairs (eigenvalue with its eigenvector) to satisfies keep_percent.
// Assumes matrix is symmetric and its size is a power of 2. Note that data is overwritten during this call.
//
// data_dim is the number of row of the matrix if it were not rounded by the next power of 2.
//
// nb_eigenpairs is the number of eigenpairs to compute (which is also the number of rows of the output matrix).
//
// epsilon is used to stop the Rayleigh iteration whenever the difference between the previous and actual
// eigenvector is smaller than that.
//
// max_rayleigh_iter is used to stop the rayleigh iteration if it does not converge anymore (precision issue?)
template <class T>
matrix_t<T> eig(queue& q, matrix_t<T>& matrix, SYCLIndexT data_dim, SYCLIndexT nb_eigenpairs = 0,
                     double epsilon = 1e-3, unsigned max_rayleigh_iter = 15) {
  matrix.assert_square();
  const auto matrix_global_range = matrix.get_range();
  const auto data_dim_pow2 = matrix_global_range[0];
  assert((data_dim_pow2 & (data_dim_pow2 - 1)) == 0 && "Matrix size must be a power of 2");
  assert(nb_eigenpairs <= data_dim && "Cannot ask for more eigenvectors than there are variables");

  if (nb_eigenpairs == 0)
    nb_eigenpairs = data_dim;
  matrix_t<T> eig_vecs(range<2>(nb_eigenpairs, data_dim_pow2));

  const auto matrix_range = get_optimal_nd_range(matrix_global_range);
  const auto index_order = ::detail::compute_index_order(q, matrix, data_dim);

  vector_t<T> act_eig_vec{range<1>(data_dim_pow2)};   // Allocated to be a power of 2
  vector_t<T> vec_buffer{range<1>(data_dim_pow2)};    // Used as a temporary vector buffer
  matrix_t<T> mat_buffer{matrix.get_range()};      // Used as a temporary matrix buffer
  matrix_t<T> c_buffer(range<2>(data_dim, 2 * data_dim));  // Temporary buffer for matrix inversion
  matrix_t<T> block_buffer(range<2>(data_dim, data_dim + 1));  // Temporary buffer for matrix inversion
  const auto actual_vec_range = get_optimal_nd_range(range<1>(data_dim));
  const auto vec_range_pow2 = get_optimal_nd_range(act_eig_vec.get_range());
  sycl_memset(q, vec_buffer, vec_range_pow2);
  sycl_memset(q, mat_buffer, matrix_range);
  T eig_val_diff;

  // Iterate to compute enough eigenpairs with Hotteling's deflation.
  // A_(i+1) = A_i - Lambda(i) * v(i) * v(i)'
  // Assuming v is normalized.
  // This deflation allow A_(i+1) to have the same eigenpairs than A_i excluding the pair (Lambda(i), v(i)).
  T eig_val;
  T eig_vals_sum = 0;
  IndexT act_rayleigh_iter;

  std::cout.precision(8);
  for (IndexT i = 0; i < nb_eigenpairs; ++i) {
    ::detail::write_initial_guess(q, matrix, act_eig_vec, index_order[i], vec_range_pow2);

    // Rayleigh quotient iteration:
    // Lambda(i) = v(i)' * A * v(i)
    // v(i+1) = (A - Lambda(i)*I)^-1 * v(i) / ||(A - Lambda(i)*I)^-1 * v(i)||
    // Note: no need to divide lambda by the norm of v if the guess is already normalized
    ::detail::rayleigh_vec_iteration(q, matrix, act_eig_vec, eig_val,
                                     vec_buffer, mat_buffer, c_buffer, block_buffer,
                                     actual_vec_range, vec_range_pow2);

    std::cout << "#" << i;
    std::cout << "\t nb_rayleigh_iter=" << act_rayleigh_iter;
    std::cout << "\t eigenvalue diff=" << std::fixed << eig_val_diff;
    std::cout << "\t eigenvalue=" << eig_val << " / " << eig_vals_sum << std::endl;

    copy_vec_to_row(q, eig_vecs, act_eig_vec, vec_range_pow2, i);
    eig_vals_sum += eig_val;

    if (i < nb_eigenpairs - 1)
      ::detail::deflate(q, matrix, act_eig_vec, eig_val, matrix_range);
  }

  return eig_vecs;
}

} // ml

#endif //INCLUDE_ML_MATH_EIG_HPP
