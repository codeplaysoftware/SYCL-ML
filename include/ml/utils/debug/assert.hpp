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
 * @brief Common assert functions, only active in debug mode.
 */

#ifndef INCLUDE_ML_UTILS_DEBUG_ASSERT_HPP
#define INCLUDE_ML_UTILS_DEBUG_ASSERT_HPP

#include <cassert>
#include <cmath>
#include <iostream>

#include "ml/utils/access.hpp"

namespace ml {

#define STATIC_ASSERT_A_IMPLIES_B(a, b) static_assert(((a) && (b)) || !(a), "")
#define STATIC_ASSERT_DATA_DIM_FOR_DIM_2(dim, d) \
  STATIC_ASSERT_A_IMPLIES_B(dim != 2, d == LIN)

#ifndef NDEBUG
template <class T>
void assert_eq(T actual, T expected) {
  if (actual != expected) {
    std::cerr << "Error: got " << actual << " expected " << expected
              << std::endl;
    assert(false);
  }
}

template <class T>
void assert_vec_eq(const T& actual, const T& expected, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    assert_eq(actual[i], expected[i]);
  }
}

template <int DIM>
void assert_rng_eq(const range<DIM>& actual, const range<DIM>& expected) {
  assert_vec_eq(actual, expected, DIM);
}

template <class T>
void assert_less_or_eq(T x, T high) {
  if (x > high) {
    std::stringstream ss;
    ss << "Error: " << x << " larger than " << high;
    std::cerr << ss.str() << std::endl;
    assert(false);
  }
}

template <int DIM>
inline void assert_rng_size_less_or_eq(const range<DIM>& r,
                                       SYCLIndexT high_size) {
  assert_less_or_eq(r.size(), high_size);
}

template <data_dim D = LIN, int DIM>
void assert_rng_less_or_eq(const range<DIM>& r, const range<DIM>& high_r) {
  for (int i = 0; i < DIM; ++i) {
    assert_less_or_eq(r[i], high_r[i]);
  }
}

template <>
inline void assert_rng_less_or_eq<TR, 2>(const range<2>& r,
                                         const range<2>& high_r) {
  assert_less_or_eq(r[1], high_r[0]);
  assert_less_or_eq(r[0], high_r[1]);
}

template <data_dim = LIN>
inline void assert_rng_less_or_eq(const range<1>& r, SYCLIndexT high0) {
  assert_rng_less_or_eq(r, range<1>(high0));
}

template <data_dim D = LIN>
inline void assert_rng_less_or_eq(const range<2>& r, SYCLIndexT high0,
                                  SYCLIndexT high1) {
  assert_rng_less_or_eq(range<2>(access_rng<D>(r, 0), access_rng<D>(r, 1)),
                        range<2>(high0, high1));
}

template <data_dim = LIN>
inline void assert_rng_less_or_eq(const range<3>& r, SYCLIndexT high0,
                                  SYCLIndexT high1, SYCLIndexT high2) {
  assert_rng_less_or_eq(r, range<3>(high0, high1, high2));
}

template <class T>
void assert_real(T x) {
  if (!std::isfinite(x)) {
    std::stringstream ss;
    ss << "Error: value is ";
    if (std::isnan(x)) {
      ss << "nan";
    } else if (std::isinf(x)) {
      ss << "inf";
    } else {
      ss << x;
    }
    std::cerr << ss.str() << std::endl;
    assert(false);
  }
}

inline void assert_rng_square(const range<2>& r) {
  assert_eq(r[0], r[1]);
}

#else   // NDEBUG
template <class T>
inline void assert_eq(T, T) {}
template <class T>
inline void assert_vec_eq(const T&, const T&, size_t) {}
template <int DIM>
inline void assert_rng_eq(const range<DIM>&, const range<DIM>&) {}
template <class T>
inline void assert_less_or_eq(T, T) {}
template <int DIM>
inline void assert_rng_size_less_or_eq(range<DIM>, SYCLIndexT) {}
template <data_dim = LIN, int DIM>
inline void assert_rng_less_or_eq(const range<DIM>&, const range<DIM>&) {}
template <data_dim = LIN>
inline void assert_rng_less_or_eq(const range<1>&, SYCLIndexT) {}
template <data_dim = LIN>
inline void assert_rng_less_or_eq(const range<2>&, SYCLIndexT, SYCLIndexT) {}
template <data_dim = LIN>
inline void assert_rng_less_or_eq(const range<3>&, SYCLIndexT, SYCLIndexT,
                                  SYCLIndexT) {}
template <class T>
inline void assert_real(T) {}
inline void assert_rng_square(const range<2>&) {}
#endif  // end NDEBUG

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_DEBUG_ASSERT_HPP
