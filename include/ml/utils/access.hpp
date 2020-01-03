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
/**
 * @file
 * @brief Define data_dim and some common functions related to it.
 */

#ifndef INCLUDE_ML_UTILS_ACCESS_HPP
#define INCLUDE_ML_UTILS_ACCESS_HPP

#include "ml/utils/sycl_types.hpp"

namespace ml {

/**
 * @brief Represent either a choice of dimension or of transposing.
 *
 * A choice of dimension means whether to use a row or a column.\n
 * A choice of transposing means whether to access the matrix as if it were
 * transposed or not.\n
 *
 */
enum data_dim {
  /// 0
  ROW = 0,
  /// Alias for ROW
  LIN = ROW,
  /// 1
  COL = 1,
  /// Alias for COL
  TR = COL
};

namespace detail {

template <class T, data_dim D>
struct lin_or_tr {
  static inline T apply(T lin, T) { return lin; }
};

template <class T>
struct lin_or_tr<T, TR> {
  static inline T apply(T, T tr) { return tr; }
};

}  // namespace detail

/**
 * @brief Return the first value if LIN, the second otherwise.
 *
 * @tparam D
 * @tparam T
 * @param lin
 * @param tr
 * @return \p lin if D=LIN, \p tr otherwise
 */
template <data_dim D, class T>
inline constexpr T lin_or_tr(T lin, T tr) {
  return detail::lin_or_tr<T, D>::apply(lin, tr);
}

/**
 * @brief Return the opposite value of D.
 *
 * @tparam D
 * @return TR if D=LIN, LIN otherwise
 */
template <data_dim D>
inline constexpr data_dim opp() {
  return static_cast<data_dim>((D + 1) % 2);
}

/**
 * @brief Access an index of a \p range<2> that may be swapped according to \p
 * D.
 *
 * @tparam D
 * @param r
 * @param i
 * @return the ith element if D=LIN, the other element otherwise
 */
template <data_dim D>
inline SYCLIndexT access_rng(const range<2>& r, SYCLIndexT i) {
  assert(i == 0 || i == 1);
  return r[lin_or_tr<D>(i, (i + 1) % 2)];
}

/**
 * @brief Construct an object \p B with the 2 given parameters that may be
 * swapped according to \p D.
 *
 * @tparam D
 * @tparam B class to build, must have a constructor with 2 @ref SYCLIndexT
 * @param x1
 * @param x2
 * @return the built object
 */
template <data_dim D, class B>
inline constexpr B build_lin_or_tr(SYCLIndexT x1, SYCLIndexT x2) {
  return B(lin_or_tr<D>(x1, x2), lin_or_tr<D>(x2, x1));
}

/**
 * @brief Construct another object \p B with the 2 parameters extracted from \p
 * b that may be swapped according to \p D.
 *
 * @see build_lin_or_tr(SYCLIndexT, SYCLIndexT)
 * @tparam D
 * @tparam B class to build, must have a constructor with 2 arguments and a
 * squared bracket accessor
 * @param b
 * @return the built object
 */
template <data_dim D, class B>
inline constexpr B build_lin_or_tr(const B& b) {
  return build_lin_or_tr<D, B>(b[0], b[1]);
}

template <data_dim D1, data_dim D2>
inline constexpr std::array<eig_dim_pair_t, 1> get_contract_dim() {
  return {eig_dim_pair_t(D1, D2)};
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_ACCESS_HPP
