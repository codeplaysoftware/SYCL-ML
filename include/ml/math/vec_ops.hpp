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
#ifndef INCLUDE_ML_MATH_VEC_OPS_HPP
#define INCLUDE_ML_MATH_VEC_OPS_HPP

#include <cmath>
#include <functional>

#include "ml/math/functors.hpp"
#include "ml/utils/buffer_t.hpp"
#include "ml/utils/copy.hpp"

namespace ml {

class ml_vec_unary_op;

/**
 * @brief out = op(in).
 *
 * @tparam UnaryOp T -> T
 * @tparam T
 * @param q
 * @param[in] in
 * @param[out] out
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <class UnaryOp, class T, int DIM>
event vec_unary_op(queue& q, buffer_t<T, DIM>& in, buffer_t<T, DIM>& out,
                   UnaryOp op = UnaryOp()) {
  return q.submit([&in, &out, op](handler& cgh) {
    auto in_acc = in.template get_access_1d<access::mode::read>(cgh);
    auto out_acc = out.template get_access_1d<access::mode::write>(cgh);
    cgh.parallel_for<NameGen<DIM, ml_vec_unary_op, T, UnaryOp>>(
        out.get_nd_range(), [=](nd_item<DIM> item) {
          auto id = item.get_global_linear_id();
          out_acc(id) = op(in_acc(id));
        });
  });
}

/**
 * @see vec_unary_op(queue&, buffer_t<T, DIM>&, buffer_t<T, DIM>&, UnaryOp)
 * @tparam UnaryOp T -> T
 * @tparam T
 * @param q
 * @param[in,out] in_out
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <class UnaryOp, class T, int DIM>
inline event vec_unary_op(queue& q, buffer_t<T, DIM>& in_out,
                          UnaryOp op = UnaryOp()) {
  return vec_unary_op(q, in_out, in_out, op);
}

class ml_vec_binary_op;

/**
 * @brief out = op(in1, in2).
 *
 * @tparam BinaryOp T -> T -> T
 * @tparam T
 * @param q
 * @param[in] in1
 * @param[in] in2
 * @param[out] out
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <class BinaryOp, class T, int DIM>
event vec_binary_op(queue& q, buffer_t<T, DIM>& in1, buffer_t<T, DIM>& in2,
                    buffer_t<T, DIM>& out, BinaryOp op = BinaryOp()) {
  return q.submit([&in1, &in2, &out, op](handler& cgh) {
    auto in1_acc = in1.template get_access_1d<access::mode::read>(cgh);
    auto in2_acc = in2.template get_access_1d<access::mode::read>(cgh);
    auto out_acc = out.template get_access_1d<access::mode::write>(cgh);
    cgh.parallel_for<NameGen<DIM, ml_vec_binary_op, T, BinaryOp>>(
        out.get_nd_range(), [=](nd_item<DIM> item) {
          auto id = item.get_global_linear_id();
          out_acc(id) = op(in1_acc(id), in2_acc(id));
        });
  });
}

/**
 * @see vec_binary_op(queue&, buffer_t<T, DIM>&, buffer_t<T, DIM>&, buffer_t<T,
 * DIM>&, BinaryOp)
 * @tparam BinaryOp T -> T -> T
 * @tparam T
 * @param q
 * @param[in, out] in_out1
 * @param[in] in2
 * @param op
 * @return A SYCL event corresponding to the submitted operation
 */
template <class BinaryOp, class T, int DIM>
inline event vec_binary_op(queue& q, buffer_t<T, DIM>& in_out1,
                           buffer_t<T, DIM>& in2, BinaryOp op = BinaryOp()) {
  return vec_binary_op(q, in_out1, in2, in_out1, op);
}

/**
 * @brief Compute the dot product.
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return <v1, v2>
 */
template <class T, int DIM>
T sycl_dot_product(queue& q, buffer_t<T, DIM>& v1, buffer_t<T, DIM>& v2) {
  auto eig_v1 = sycl_to_eigen(v1);
  auto eig_v2 = sycl_to_eigen(v2);
  vector_t<T> out(range<1>(1));
  auto eig_out = sycl_to_eigen<1, 0>(out);
  eig_out.device() = (eig_v1.tensor() * eig_v2.tensor()).sum();
  T dot;
  auto event = sycl_copy_device_to_host(q, out, &dot);
  event.wait_and_throw();
  return dot;
}

/**
 * @brief Compute the dot product.
 *
 * @see sycl_dot_product(queue&, buffer_t<T, DIM>&, buffer_t<T, DIM>&)
 * @tparam T
 * @param q
 * @param v
 * @return <v, v>
 */
template <class T, int DIM>
inline T sycl_dot_product(queue& q, buffer_t<T, DIM>& v) {
  return sycl_dot_product(q, v, v);
}

/**
 * @brief Computes the norm of v.
 *
 * @tparam T
 * @param q
 * @param v
 * @return ||v||
 */
template <class T, int DIM>
T sycl_norm(queue& q, buffer_t<T, DIM>& v) {
  return std::sqrt(sycl_dot_product(q, v));
}

/**
 * @brief v /= norm.
 *
 * @tparam T
 * @param q
 * @param v
 * @param norm
 * @return A SYCL event corresponding to the submitted operation
 */
template <class T, int DIM>
inline event sycl_normalize(queue& q, buffer_t<T, DIM>& v, T norm) {
  return vec_unary_op(
      q, v, functors::partial_binary_op<T, std::multiplies<T>>(1 / norm));
}

/**
 * @brief v /= ||v||.
 *
 * @tparam T
 * @param q
 * @param v
 * @return A SYCL event corresponding to the submitted operation
 */
template <class T, int DIM>
inline event sycl_normalize(queue& q, buffer_t<T, DIM>& v) {
  return sycl_normalize(q, v, sycl_norm(q, v));
}

/**
 * @brief Distance between v1 and v2 squared.
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return ||v1 - v2||^2
 */
template <class T, int DIM>
T sycl_dist2(queue& q, buffer_t<T, DIM>& v1, buffer_t<T, DIM>& v2) {
  auto eig_v1 = sycl_to_eigen(v1);
  auto eig_v2 = sycl_to_eigen(v2);
  vector_t<T> out(range<1>(1));
  auto eig_out = sycl_to_eigen<1, 0>(out);
  auto cwise_diff = eig_v1.tensor() - eig_v2.tensor();
  eig_out.device() = (cwise_diff * cwise_diff).sum();
  T dist2;
  auto event = sycl_copy_device_to_host(q, out, &dist2);
  event.wait_and_throw();
  return dist2;
}

/**
 * @brief Distance between v1 and -v2 squared.
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return ||v1 + v2||^2
 */
template <class T, int DIM>
T sycl_dist2_opposite(queue& q, buffer_t<T, DIM>& v1, buffer_t<T, DIM>& v2) {
  auto eig_v1 = sycl_to_eigen(v1);
  auto eig_v2 = sycl_to_eigen(v2);
  vector_t<T> out(range<1>(1));
  auto eig_out = sycl_to_eigen<1, 0>(out);
  auto cwise_sum = eig_v1.tensor() + eig_v2.tensor();
  eig_out.device() = (cwise_sum * cwise_sum).sum();
  T dist2;
  auto event = sycl_copy_device_to_host(q, out, &dist2);
  event.wait_and_throw();
  return dist2;
}

/**
 * @brief Distance between v1 and v2.
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return ||v1 - v2||
 */
template <class T, int DIM>
inline T sycl_dist(queue& q, buffer_t<T, DIM>& v1, buffer_t<T, DIM>& v2) {
  return std::sqrt(sycl_dist2(q, v1, v2));
}

/**
 * @brief Computes the distance between v1 and v2 ignoring their direction.
 *
 * This is useful when comparing eigenvectors which direction does not matter
 * for instance.
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return min(||v1 - v2||, ||v1 + v2||)
 */
template <class T, int DIM>
inline T sycl_dist_no_direction(queue& q, buffer_t<T, DIM>& v1,
                                buffer_t<T, DIM>& v2) {
  return std::sqrt(
      std::min(sycl_dist2(q, v1, v2), sycl_dist2_opposite(q, v1, v2)));
}

/**
 * @brief Min of v.
 *
 * @tparam T
 * @param q
 * @param v
 * @return min(v)
 */
template <class T, int DIM>
T sycl_min(queue& q, buffer_t<T, DIM>& v) {
  auto eig_v = sycl_to_eigen(v);
  vector_t<T> out(range<1>(1));
  auto eig_out = sycl_to_eigen<1, 0>(out);
  eig_out.device() = eig_v.tensor().minimum();
  T min;
  auto event = sycl_copy_device_to_host(q, out, &min);
  event.wait_and_throw();
  return min;
}

/**
 * @brief Max of v.
 *
 * @tparam T
 * @param q
 * @param v
 * @return max(v)
 */
template <class T, int DIM>
T sycl_max(queue& q, buffer_t<T, DIM>& v) {
  auto eig_v = sycl_to_eigen(v);
  vector_t<T> out(range<1>(1));
  auto eig_out = sycl_to_eigen<1, 0>(out);
  eig_out.device() = eig_v.tensor().maximum();
  T max;
  auto event = sycl_copy_device_to_host(q, out, &max);
  event.wait_and_throw();
  return max;
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_VEC_OPS_HPP
