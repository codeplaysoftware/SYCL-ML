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
#ifndef INCLUDE_ML_MATH_MAT_INV_HPP
#define INCLUDE_ML_MATH_MAT_INV_HPP

#include "ml/math/mat_mul.hpp"
#include "ml/math/mat_ops.hpp"

namespace ml {

class ml_mat_inv;

/**
 * @brief Invert the given matrix of size nxn.
 *
 * Uses the Gauss-Jordan method.
 *
 * @see tri_solve(queue&, matrix_t<T>&, matrix_t<T>&) for a more numerically
 * stable solution
 * @tparam T
 * @param q
 * @param[in] mat
 * @param[out] inv
 * @param c_buffer temporary buffer must be at least of size nx(2*n)
 * @param block_buffer temporary buffer must be at least of size nx(n+1)
 * @return A SYCL event corresponding to the last submitted operation
 */
template <class T>
event mat_inv(queue& q, matrix_t<T>& mat, matrix_t<T>& inv,
              matrix_t<T>& c_buffer, matrix_t<T>& block_buffer) {
  auto data_dim = mat.data_range[1];
  mat.assert_square();
  assert_rng_less_or_eq(mat.get_kernel_range(), inv.data_range);
  assert_rng_less_or_eq(c_buffer.data_range, data_dim, 2 * data_dim);
  assert_rng_less_or_eq(block_buffer.data_range, data_dim, data_dim + 1);

  // C = [A|I]
  q.submit([&mat, &c_buffer](handler& cgh) {
    auto mat_acc = mat.template get_access_2d<access::mode::read>(cgh);
    auto c_acc =
        c_buffer.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_mat_inv, T>>(
        c_buffer.get_nd_range(), [=](nd_item<2> item) {
          auto global_nb_rows = item.get_global_range()[0];
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          // Copy A if left part, set identity otherwise
          c_acc(row, col) = (col < global_nb_rows)
                                ? mat_acc(row, col)
                                : ((row + global_nb_rows) == col);
        });
  });

  // Compute C so that C = [I|A^-1]
  for (decltype(data_dim) r = 0; r < data_dim; ++r) {
    // Write update in block_buffer
    q.submit([&c_buffer, &block_buffer, r](handler& cgh) {
      auto c_acc = c_buffer.template get_access_2d<access::mode::read>(cgh);
      auto block_acc =
          block_buffer.template get_access_2d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGen<1, ml_mat_inv, T>>(
          block_buffer.get_nd_range(), [=](nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            int is_row_eq_r = row == r;
            // if row == r: C(i,j) = C(i,j) / C(r,r)
            // else:        C(i,j) = C(i,j) - (C(i,r) / C(r,r)) * C(r, j)
            block_acc(row, col) =
                is_row_eq_r * (c_acc(row, col + r) / c_acc(r, r)) +
                !is_row_eq_r *
                    (c_acc(row, col + r) -
                     (c_acc(row, r) / c_acc(r, r)) * c_acc(r, col + r));
          });
    });

    // Copy block_buffer in c_buffer
    q.submit([&c_buffer, &block_buffer, r](handler& cgh) {
      auto c_acc = c_buffer.template get_access_2d<access::mode::write>(cgh);
      auto block_acc =
          block_buffer.template get_access_2d<access::mode::read>(cgh);
      cgh.parallel_for<NameGen<2, ml_mat_inv, T>>(
          block_buffer.get_nd_range(), [=](nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            c_acc(row, col + r) = block_acc(row, col);
          });
    });
  }

  // Copy the right part of C to inv
  return q.submit([&c_buffer, &inv](handler& cgh) {
    auto c_acc = c_buffer.template get_access_2d<access::mode::read>(cgh);
    auto inv_acc = inv.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<3, ml_mat_inv, T>>(
        inv.get_nd_range(), [=](nd_item<2> item) {
          auto global_nb_rows = item.get_global_range()[0];
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          inv_acc(row, col) = c_acc(row, global_nb_rows + col);
        });
  });
}

/**
 * @brief Invert the given matrix and create any necessary temporary buffers.
 *
 * @see mat_inv(queue&, matrix_t<T>&, matrix_t<T>&, matrix_t<T>&, matrix_t<T>&)
 * @tparam T
 * @param q
 * @param[in] mat
 * @param[out] inv
 * @return A SYCL event corresponding to the last submitted operation
 */
template <class T>
event mat_inv(queue& q, matrix_t<T>& mat, matrix_t<T>& inv) {
  auto data_dim = mat.data_range[1];
  matrix_t<T> c_buffer(range<2>(data_dim, 2 * data_dim));
  matrix_t<T> block_buffer(range<2>(data_dim, data_dim + 1));
  return mat_inv(q, mat, inv, c_buffer, block_buffer);
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_MAT_INV_HPP
