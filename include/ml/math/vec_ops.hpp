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
 */
template <class UnaryOp, class T>
void vec_unary_op(queue& q, sycl_vec_t<T>& in, sycl_vec_t<T>& out,
                  UnaryOp op = UnaryOp()) {
  sycl::sycl_execution_policy<NameGen<0, ml_vec_unary_op, T, UnaryOp>>
      sycl_policy(q);
  transform(sycl_policy, begin(in), end(in), begin(out), op);
}

/**
 * @see vec_unary_op(queue&, sycl_vec_t<T>&, sycl_vec_t<T>&, UnaryOp)
 * @tparam UnaryOp T -> T
 * @tparam T
 * @param q
 * @param[in,out] in_out
 * @param op
 */
template <class UnaryOp, class T>
inline void vec_unary_op(queue& q, sycl_vec_t<T>& in_out,
                         UnaryOp op = UnaryOp()) {
  vec_unary_op(q, in_out, in_out, op);
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
 */
template <class BinaryOp, class T>
void vec_binary_op(queue& q, sycl_vec_t<T>& in1, sycl_vec_t<T>& in2,
                   sycl_vec_t<T>& out, BinaryOp op = BinaryOp()) {
  sycl::sycl_execution_policy<NameGen<0, ml_vec_binary_op, T, BinaryOp>>
      sycl_policy(q);
  transform(sycl_policy, begin(in1), end(in1), begin(in2), begin(out), op);
}

/**
 * @see vec_binary_op(queue&, sycl_vec_t<T>&, sycl_vec_t<T>&, sycl_vec_t<T>&,
 * BinaryOp)
 * @tparam BinaryOp T -> T -> T
 * @tparam T
 * @param q
 * @param[in, out] in_out1
 * @param[in] in2
 * @param op
 */
template <class BinaryOp, class T>
inline void vec_binary_op(queue& q, sycl_vec_t<T>& in_out1, sycl_vec_t<T>& in2,
                          BinaryOp op = BinaryOp()) {
  vec_binary_op(q, in_out1, in2, in_out1, op);
}

class ml_inner_prod;

/**
 * @brief Compute the inner product (or dot product).
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return <v1, v2>
 */
template <class T>
T sycl_inner_product(queue& q, sycl_vec_t<T>& v1, sycl_vec_t<T>& v2) {
  sycl::sycl_execution_policy<NameGen<0, ml_inner_prod, T>> sycl_policy(q);
  return inner_product(sycl_policy, begin(v1), end(v1), begin(v2), 0.0f);
}

/**
 * @brief Compute the inner product (or dot product).
 *
 * @see sycl_inner_product(queue&, sycl_vec_t<T>&, sycl_vec_t<T>&)
 * @tparam T
 * @param q
 * @param v
 * @return <v, v>
 */
template <class T>
inline T sycl_inner_product(queue& q, sycl_vec_t<T>& v) {
  return sycl_inner_product(q, v, v);
}

/**
 * @brief Computes the norm of v.
 *
 * @tparam T
 * @param q
 * @param v
 * @return ||v||
 */
template <class T>
T sycl_norm(queue& q, sycl_vec_t<T>& v) {
  return std::sqrt(sycl_inner_product(q, v));
}

class ml_transform;

/**
 * @brief v = op(v, cst).
 *
 * @tparam BinaryOp T -> T -> T
 * @tparam T
 * @param q
 * @param v
 * @param cst
 * @param op
 */
template <class BinaryOp, class T>
void sycl_transform(queue& q, sycl_vec_t<T>& v, T cst,
                    BinaryOp op = BinaryOp()) {
  sycl::sycl_execution_policy<NameGen<0, ml_transform, T, BinaryOp>>
      sycl_policy(q);
  transform(sycl_policy, begin(v), end(v), begin(v),
            [=](T val) { return op(val, cst); });
}

/**
 * @brief v /= norm.
 *
 * @tparam T
 * @param q
 * @param v
 * @param norm
 */
template <class T>
inline void sycl_normalize(queue& q, sycl_vec_t<T>& v, T norm) {
  sycl_transform(q, v, norm, std::divides<T>());
}

/**
 * @brief v /= ||v||.
 *
 * @tparam T
 * @param q
 * @param v
 */
template <class T>
inline void sycl_normalize(queue& q, sycl_vec_t<T>& v) {
  sycl_normalize(q, v, sycl_norm(q, v));
}

class ml_dist2;

/**
 * @brief Distance between v1 and v2 squared.
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return ||v1 - v2||^2
 */
template <class T>
T sycl_dist2(queue& q, sycl_vec_t<T>& v1, sycl_vec_t<T>& v2) {
  sycl::sycl_execution_policy<NameGen<0, ml_dist2, T>> sycl_policy(q);
  return inner_product(sycl_policy, begin(v1), end(v1), begin(v2), 0.0f,
                       std::plus<T>(), functors::prod_diff<T>());
}

class ml_dist2_opposite;

/**
 * @brief Distance between v1 and -v2 squared.
 *
 * @tparam T
 * @param q
 * @param v1
 * @param v2
 * @return ||v1 + v2||^2
 */
template <class T>
T sycl_dist2_opposite(queue& q, sycl_vec_t<T>& v1, sycl_vec_t<T>& v2) {
  sycl::sycl_execution_policy<NameGen<0, ml_dist2_opposite, T>> sycl_policy(q);
  return inner_product(sycl_policy, begin(v1), end(v1), begin(v2), 0.0f,
                       std::plus<T>(), functors::prod_sum<T>());
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
template <class T>
inline T sycl_dist(queue& q, sycl_vec_t<T>& v1, sycl_vec_t<T>& v2) {
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
template <class T>
inline T sycl_dist_no_direction(queue& q, sycl_vec_t<T>& v1,
                                sycl_vec_t<T>& v2) {
  return std::sqrt(
      std::min(sycl_dist2(q, v1, v2), sycl_dist2_opposite(q, v1, v2)));
}

class ml_min;

/**
 * @brief Min of v.
 *
 * @tparam T
 * @param q
 * @param v
 * @return min(v)
 */
template <class T>
T sycl_min(queue& q, sycl_vec_t<T>& v) {
  sycl::sycl_execution_policy<NameGen<0, ml_min, T>> sycl_policy(q);
  return reduce(sycl_policy, begin(v), end(v), std::numeric_limits<T>::max(),
                [](T a, T b) { return a < b ? a : b; });
}

class ml_max;

/**
 * @brief Max of v.
 *
 * @tparam T
 * @param q
 * @param v
 * @return max(v)
 */
template <class T>
T sycl_max(queue& q, sycl_vec_t<T>& v) {
  sycl::sycl_execution_policy<NameGen<0, ml_max, T>> sycl_policy(q);
  return reduce(sycl_policy, begin(v), end(v), std::numeric_limits<T>::min(),
                [](T a, T b) { return a > b ? a : b; });
}

}  // namespace ml

#endif  // INCLUDE_ML_MATH_VEC_OPS_HPP
