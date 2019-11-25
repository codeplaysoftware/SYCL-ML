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
#ifndef INCLUDE_ML_MATH_TRI_INV_HPP
#define INCLUDE_ML_MATH_TRI_INV_HPP

#include "ml/utils/common.hpp"

namespace ml {

class ml_try_inv;

/**
 * @brief Invert the given upper triangular matrix of size nxn.
 *
 * Uses the Gauss-Jordan method.
 *
 * @see tri_solve(queue&, matrix_t<T>&, matrix_t<T>&) for a more numerically
 * stable solution
 * @tparam T
 * @param q
 * @param[in] tri
 * @param[out] inv
 * @param t_buffer temporary buffer must be of size nxn at least.
 * @param t_pow_buffer temporary buffer must be of size nxn at least.
 * @param data_dim_1_nd_rng 1d kernel range of size n.
 * @return A SYCL event corresponding to the last submitted operation
 */
template <class T>
event tri_inv(queue& q, matrix_t<T>& tri, matrix_t<T>& inv,
              matrix_t<T>& t_buffer, matrix_t<T>& t_pow_buffer,
              const nd_range<1>& data_dim_1_nd_rng) {
  assert(&tri != &inv);
  assert(&tri != &t_buffer);
  assert(&tri != &t_pow_buffer);
  assert(&inv != &t_buffer);
  assert(&inv != &t_pow_buffer);
  assert(&t_buffer != &t_pow_buffer);

  auto data_dim_2_rng = tri.kernel_range.get_global_range();
  assert_eq(data_dim_1_nd_rng.get_global_range()[0], data_dim_2_rng[0]);
  assert_rng_square(data_dim_2_rng);
  auto data_dim = data_dim_2_rng[0];
  using IndexT = decltype(data_dim);
  assert_rng_less_or_eq(data_dim_2_rng, tri.data_range);
  assert_rng_less_or_eq(data_dim_2_rng, inv.data_range);
  assert_rng_less_or_eq(data_dim_2_rng, t_buffer.data_range);
  assert_rng_less_or_eq(data_dim_2_rng, t_pow_buffer.data_range);

  q.submit([&tri, &t_buffer, &t_pow_buffer, &inv](handler& cgh) {
    auto tri_acc = tri.template get_access_2d<access::mode::read>(cgh);
    auto t_acc =
        t_buffer.template get_access_2d<access::mode::discard_write>(cgh);
    auto t_pow_acc =
        t_pow_buffer.template get_access_2d<access::mode::discard_write>(cgh);
    auto inv_acc = inv.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_try_inv, T>>(
        tri.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          T val = (col > row) ? (-tri_acc(row, col) / tri_acc(row, row)) : 0;
          t_acc(row, col) = val;
          t_pow_acc(row, col) = val;
          inv_acc(row, col) = (row == col) ? 1 : val;
        });
  });

  auto tri_nd_range = tri.get_nd_range();
  for (IndexT i = 2; i < data_dim; ++i) {  // i = 0 -> id; i = 1 -> t_acc
    // mat_mul where we know some zeros
    q.submit([&t_pow_buffer, &t_buffer, &inv, tri_nd_range, data_dim,
              i](handler& cgh) {
      auto t_pow_acc =
          t_pow_buffer.template get_access_2d<access::mode::read_write>(cgh);
      auto t_acc = t_buffer.template get_access_2d<access::mode::read>(cgh);
      auto inv_acc = inv.template get_access_2d<access::mode::read_write>(cgh);
      cgh.parallel_for<NameGen<2, ml_try_inv, T>>(
          tri_nd_range, [=](nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            if (row < data_dim - i && col < data_dim - i && col >= row) {
              auto diag_idx = col - row;
              col += i;
              T sum = 0;
              // don't use the full line or column because of zeros
              for (size_t j = 0; j <= diag_idx; ++j) {
                sum += t_pow_acc(row, row + i + j - 1) *
                       t_acc(row + i + j - 1, col);
              }
              // Store the result in the lower triangle part and transpose it
              // later
              t_pow_acc(col, row) = sum;
              inv_acc(row, col) += sum;
            }
          });
    });

    // Transpose lower part of t_pow_acc to upper part
    q.submit([&t_pow_buffer, tri_nd_range, data_dim, i](handler& cgh) {
      auto t_pow_acc =
          t_pow_buffer.template get_access_2d<access::mode::read_write>(cgh);
      cgh.parallel_for<NameGen<3, ml_try_inv, T>>(
          tri_nd_range, [=](nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            if (row < data_dim - i && col < data_dim - i && col >= row) {
              col += i;
              t_pow_acc(row, col) = t_pow_acc(col, row);
            }
          });
    });
  }

  return q.submit([&tri, &inv](handler& cgh) {
    auto tri_acc = tri.template get_access_2d<access::mode::read>(cgh);
    auto inv_acc = inv.template get_access_2d<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<4, ml_try_inv, T>>(
        tri.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          inv_acc(row, col) /= tri_acc(col, col);
        });
  });
}

/**
 * @brief Invert the given upper triangular matrix and create any necessary
 * temporary buffers.
 *
 * @see tri_inv(queue&, matrix_t<T>&, matrix_t<T>&, matrix_t<T>&, matrix_t<T>&,
 * const nd_range<1>&)
 * @tparam T
 * @param q
 * @param[in] tri
 * @param[out] inv
 * @return A SYCL event corresponding to the last submitted operation
 */
template <class T>
event tri_inv(queue& q, matrix_t<T>& tri, matrix_t<T>& inv) {
  tri.assert_square();
  assert_rng_eq(tri.get_kernel_range(), inv.get_kernel_range());

  matrix_t<T> t_buffer{tri.data_range, tri.kernel_range};
  matrix_t<T> t_pow_buffer{tri.data_range, tri.kernel_range};

  return tri_inv(q, tri, inv, t_buffer, t_pow_buffer,
                 get_optimal_nd_range(tri.data_range[0]));
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_TRI_INV_HPP
