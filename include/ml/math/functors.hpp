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
#ifndef INCLUDE_ML_MATH_FUNCTORS_HPP
#define INCLUDE_ML_MATH_FUNCTORS_HPP

#include "ml/utils/common.hpp"

namespace ml {

namespace functors {

template <class T>
struct positive {
  constexpr T operator()(T x) const { return x > 0; }
};

template <class T>
struct negative {
  constexpr T operator()(T x) const { return x < 0; }
};

template <class T>
struct identity {
  constexpr T operator()(T x) const { return x; }
};

template <class T>
struct sqrt {
  constexpr T operator()(T x) const { return cl::sycl::sqrt(x); }
};

template <class T, class BinaryOp>
class partial_binary_op {
 public:
  partial_binary_op(T c, BinaryOp binary_op = BinaryOp())
      : _c(c), _binary_op(binary_op) {}

  inline constexpr T operator()(T x) const { return _binary_op(_c, x); }

 private:
  T _c;
  BinaryOp _binary_op;
};

template <class T>
struct sum_log_abs {
  inline constexpr T operator()(T x1, T x2) const {
    return x1 + cl::sycl::log(cl::sycl::fabs(x2));
  }
};

template <class T>
struct exp_diff {
  template <class T1, class T2>
  constexpr T operator()(T1 x1, T2 x2) const {
    return cl::sycl::exp(x1 - x2);
  }
};

template <class T>
struct amortize {
  amortize(T factor) : _factor(factor) {}
  constexpr T operator()(T act, T prev) const { return act - prev * _factor; }

 private:
  T _factor;
};

}  // namespace functors

}  // namespace ml

#endif  // INCLUDE_ML_MATH_FUNCTORS_HPP
