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
#ifndef INCLUDE_ML_MATH_MAT_OPS_HPP
#define INCLUDE_ML_MATH_MAT_OPS_HPP

#include "ml/utils/common.hpp"

namespace ml {

class ml_eye;

/**
 * @brief Writes the identity matrix in mat
 * @tparam T
 * @param q
 * @param[out] mat
 * @return A SYCL event corresponding to the submitted operation
 */
template <class T>
event eye(queue& q, matrix_t<T>& mat) {
  mat.assert_square();

  return q.submit([&mat](handler& cgh) {
    auto mat_acc = mat.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_eye, T>>(mat.get_nd_range(),
                                            [=](nd_item<2> item) {
                                              auto row = item.get_global_id(0);
                                              auto col = item.get_global_id(1);
                                              mat_acc(row, col) = row == col;
                                            });
  });
}

class ml_transpose;

/**
 * @brief out = in'.
 *
 * @tparam T
 * @param q
 * @param[in] in
 * @param[out] out
 * @return A SYCL event corresponding to the submitted operation
 */
template <class T>
event transpose(queue& q, matrix_t<T>& in, matrix_t<T>& out) {
  assert_eq(in.get_kernel_range()[0], out.get_kernel_range()[1]);
  assert_eq(in.get_kernel_range()[1], out.get_kernel_range()[0]);
  assert(&in != &out);

  return q.submit([&in, &out](handler& cgh) {
    auto in_acc = in.template get_access_2d<access::mode::read>(cgh);
    auto out_acc = out.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_transpose, T>>(
        in.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          out_acc(col, row) = in_acc(row, col);
        });
  });
}

class ml_mat_inplace_binary_op;

/**
 * @brief in_out1 = op(in_out1, in2).
 *
 * @tparam D1 LIN or TR in_out1
 * @tparam D2 LIN or TR in2
 * @tparam T
 * @tparam BinaryOp T -> T -> T
 * @param q
 * @param[in, out] in_out1
 * @param[in] in2
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <data_dim D1 = LIN, data_dim D2 = LIN, class T, class BinaryOp>
event mat_inplace_binary_op(queue& q, matrix_t<T>& in_out1, matrix_t<T>& in2,
                            BinaryOp op = BinaryOp()) {
  assert_eq(access_ker_dim<D1>(in_out1, 0), access_ker_dim<D2>(in2, 0));
  assert_eq(access_ker_dim<D1>(in_out1, 1), access_ker_dim<D2>(in2, 1));

  return q.submit([&in_out1, &in2, op](handler& cgh) {
    auto in_out1_acc =
        in_out1.template get_access_2d<access::mode::read_write, D1>(cgh);
    auto in2_acc = in2.template get_access_2d<access::mode::read, D2>(cgh);
    cgh.parallel_for<
        NameGen<D1 * 2 + D2, ml_mat_inplace_binary_op, T, BinaryOp>>(
        in_out1.template get_nd_range<D1>(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          in_out1_acc(row, col) = op(in_out1_acc(row, col), in2_acc(row, col));
        });
  });
}

/**
 * @brief Compute the average of a specific matrix dimension.
 *
 * @tparam D dimension to reduce
 * @tparam T
 * @param[in] dataset
 * @param[out] avg
 */
template <data_dim D = LIN, class T>
void avg(queue&, matrix_t<T>& dataset, vector_t<T>& avg) {
  auto eig_dataset = sycl_to_eigen(dataset);
  auto eig_avg = sycl_to_eigen(avg);

  static const std::array<eig_index_t, 1> dims{D};
  eig_avg.device() = eig_dataset.tensor().sum(dims) /
                     static_cast<T>(access_data_dim<D>(dataset, 0));
}

class ml_mat_vec_binary_op;

/**
 * @brief out = op(in, vec).
 *
 * @tparam D whether to apply the operator for each row or each column
 * @tparam T
 * @tparam BinaryOp T -> T -> T
 * @param q
 * @param[in] in
 * @param[out] out
 * @param[in] vec
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <data_dim D = ROW, class T, class BinaryOp>
event mat_vec_apply_op(queue& q, matrix_t<T>& in, matrix_t<T>& out,
                       vector_t<T>& vec, BinaryOp op = BinaryOp()) {
  assert_rng_less_or_eq(out.get_kernel_range(), in.get_kernel_range());
  assert_less_or_eq(access_ker_dim<D>(out, 0), vec.get_kernel_size());

  return q.submit([&in, &out, &vec, op](handler& cgh) {
    auto vec_acc = vec.template get_access_1d<access::mode::read>(cgh);
    auto in_acc = in.template get_access_2d<access::mode::read>(cgh);
    auto out_acc = out.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<D, ml_mat_vec_binary_op, T, BinaryOp>>(
        out.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          out_acc(row, col) =
              op(in_acc(row, col), vec_acc(lin_or_tr<D>(row, col)));
        });
  });
}

class ml_mat_vec_inplace_binary_op;

/**
 * @brief in_out = op(in_out, vec).
 *
 * @see mat_vec_apply_op(queue&, matrix_t<T>&, matrix_t<T>&, vector_t<T>&,
 * BinaryOp)
 * @tparam D whether to apply the operator for each row or each column
 * @tparam T
 * @tparam BinaryOp T -> T -> T
 * @param q
 * @param[in, out] in_out
 * @param[in] vec
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <data_dim D = ROW, class T, class BinaryOp>
event mat_vec_apply_op(queue& q, matrix_t<T>& in_out, vector_t<T>& vec,
                       BinaryOp op = BinaryOp()) {
  assert_less_or_eq(access_ker_dim<D>(in_out, 0), vec.get_kernel_size());

  return q.submit([&vec, &in_out, op](handler& cgh) {
    auto vec_acc = vec.template get_access_1d<access::mode::read>(cgh);
    auto in_out_acc =
        in_out.template get_access_2d<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<D, ml_mat_vec_inplace_binary_op, T, BinaryOp>>(
        in_out.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          in_out_acc(row, col) =
              op(in_out_acc(row, col), vec_acc(lin_or_tr<D>(row, col)));
        });
  });
}

class ml_mat_vec_binary_op_data_rng;

/**
 * @brief Similar to mat_vec_apply_op(queue&, matrix_t<T>&, vector_t<T>&,
 * BinaryOp) except it applies the operator on the data_range only.
 * It is potentially slower because of the extra if but is useful to apply
 * an operator on a smaller range.
 *
 * @tparam D whether to apply the operator for each row or each column
 * @tparam T
 * @tparam BinaryOp T -> T -> T
 * @param q
 * @param[in, out] in_out
 * @param[in] vec
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <data_dim D = ROW, class T, class BinaryOp>
event mat_vec_apply_op_data_rng(queue& q, matrix_t<T>& in_out, vector_t<T>& vec,
                                BinaryOp op = BinaryOp()) {
  assert_less_or_eq(access_ker_dim<D>(in_out, 0), vec.get_kernel_size());

  auto data_dim_0 = access_data_dim(in_out, 0);
  auto data_dim_1 = access_data_dim(in_out, 1);
  return q.submit([&in_out, &vec, data_dim_0, data_dim_1, op](handler& cgh) {
    auto vec_acc = vec.template get_access_1d<access::mode::read>(cgh);
    auto mat_acc = in_out.template get_access_2d<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<D, ml_mat_vec_binary_op_data_rng, T, BinaryOp>>(
        in_out.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          if (row < data_dim_0 && col < data_dim_1) {
            mat_acc(row, col) =
                op(mat_acc(row, col), vec_acc(lin_or_tr<D>(row, col)));
          }
        });
  });
}

/**
 * @brief Substract the average for each rows (resp. columns).
 *
 * @tparam D Substract on rows or columns
 * @tparam T
 * @param q
 * @param[in, out] dataset
 * @param[in] data_avg
 * @return A SYCL event corresponding to the submitted operation
 */
template <data_dim D = ROW, class T>
inline event center_data(queue& q, matrix_t<T>& dataset,
                         vector_t<T>& data_avg) {
  return mat_vec_apply_op<D>(q, dataset, data_avg, std::minus<T>());
}

class ml_reduce_diag;

/**
 * @brief Apply a reduction on the diagonal of a matrix.
 *
 * @tparam Reduce T -> T -> T
 * @tparam T
 * @param q
 * @param[in] mat
 * @param offset use the main diagonal if 0, an upper diagonal if offset > 0, a
 * lower diagonal if offset < 0
 * @param init first value given as the first argument to the reduce
 * @param reduce
 * @return result of the reduce
 */
template <class Reduce, class T>
T reduce_diag(queue& q, matrix_t<T>& mat, long offset = 0, T init = 0,
              Reduce reduce = Reduce()) {
  auto diag_len = access_data_dim(mat, 1) - std::abs(offset);
  SYCLIndexT row_offset = offset < 0 ? -offset : 0;
  SYCLIndexT col_offset = offset > 0 ? offset : 0;
  {
    sycl_vec_t<T> out(&init, range<1>(1));
    q.submit([&](handler& cgh) {
      auto mat_acc = mat.template get_access_2d<access::mode::read>(cgh);
      auto out_acc = out.template get_access<access::mode::read_write>(cgh);
      cgh.single_task<NameGen<0, ml_reduce_diag, T, Reduce>>([=]() {
        for (SYCLIndexT i = 0; i < diag_len; ++i) {
          out_acc[0] =
              reduce(out_acc[0], mat_acc(i + row_offset, i + col_offset));
        }
      });
    });
  }
  return init;
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_MAT_OPS_HPP
